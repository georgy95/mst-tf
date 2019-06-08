import glob
import os
import gc
import numpy as np
from PIL import Image
from random import choice

from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.python.lib.io import file_io
from tensorflow.io.gfile import listdir
import tensorflow as tf

from traceback import format_exc

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, cosine_distances
from maxflow.fastmin import aexpansion_grid

from sys import platform as sys_pf

# matplotlib for macos hack
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def get_Fcs(Fc, Fs, k=3, alpha=0.5):

    vgg_feature_h = Fc.shape[1]
    vgg_feature_w = Fc.shape[2]

    Fs = Fs.reshape((-1, 512)).astype(np.float64) 
    Fc = Fc.reshape((-1, 512)).astype(np.float64)


    def apply_kmeans(Fs, k):
        KM = KMeans(n_clusters=k, random_state=0, max_iter=1000)
        KM.fit(Fs)
        return KM

    def get_style_feature_map(KM):
        return KM.labels_
    
    def get_content_feature_map(Fc, KM, k, l=0.3):
        V = np.ones((k, )) * l - np.identity(k) * l
        D = np.stack([cosine_distances(Fc, KM.cluster_centers_[i].reshape((1, -1))) for i in range(k)], axis=-1).reshape((vgg_feature_h, vgg_feature_w, -1)).astype(np.double)
        content_labels = aexpansion_grid(D, V, max_cycles=None, labels=None)
        return content_labels


    def calc_Fcs(Fc, Fs, Lc, Ls, a, k, epsilon=1e-5):
        
        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) >= -epsilon)

        # create a dummy array
        Fcs = Fc.copy() # (1024, 512)

        # for each cluster, get corresponding Fc and Fs and update them
        for j in range(k):
            # get style and content indices
            style_indices = np.where(Ls.reshape((-1,)) == j)[0]
            content_indices = np.where(Lc.reshape((-1,)) == j)[0]

            if len(content_indices) == 0:
                # No content features found with this label, skipping to next label
                continue

            # get this subset from all style features
            # (n, 512)
            fs = Fs[style_indices, :]
            fc = Fc[content_indices, :]

            # reshape to HWxC -> CxHW
            fs = np.transpose(fs, (1, 0))
            fc = np.transpose(fc, (1, 0))

            # for centering
            ms = np.mean(fs, axis=1, keepdims=True) # Cx1
            mc = np.mean(fc, axis=1, keepdims=True) # Cx1

            # Center content and style features CxWH
            fs = fs - ms 
            fc = fc - mc 

            # Start Whitening and Colouring Procedure
            # Read more https://arxiv.org/pdf/1904.04443v1.pdf
            

            # Content
            fcfc = np.dot(fc, fc.T) / (fc.shape[1] - 1) # CxC
            assert is_pos_def(fcfc), 'Must be +ve definite'
            Ec, wc, _ = np.linalg.svd(fcfc) # CxC
            k_c = (wc > 1e-5).sum()
            Dc = np.diag((wc[:k_c]+epsilon)**-0.5)
            fc_hat = Ec[:,:k_c].dot(Dc).dot(Ec[:,:k_c].T).dot(fc)

            # Style
            fsfs = np.dot(fs, fs.T) / (fs.shape[1] - 1)
            assert is_pos_def(fsfs), 'Must be +ve definite'
            Es, ws, _ = np.linalg.svd(fsfs)
            k_s = (ws > 1e-5).sum()
            Ds = np.sqrt(np.diag(ws[:k_s]+epsilon))

            fcs_hat = Es[:,:k_s].dot(Ds).dot(Es[:,:k_s].T).dot(fc_hat)

            # Re-centre # CxHW
            fcs_hat = fcs_hat + ms

            # Reshape Fcs_k CxHW -> HWxC
            fcs_hat = np.transpose(fcs_hat, (1,0))

            # Update specific content-style features
            Fcs[content_indices, :] = fcs_hat

        Fcs = a * Fcs + (1 - a) * Fc

        return Fcs

    KM = apply_kmeans(Fs, k)
    Ls = get_style_feature_map(KM)
    Lc = get_content_feature_map(Fc, KM, k)
    Fcs = calc_Fcs(Fc, Fs, Lc, Ls, a=alpha, k=k).reshape((vgg_feature_h, vgg_feature_w, 512))

    return Fcs


class DataLoader(object):
    def __init__(self, datapath):
        """        
        :param string datapath: filepath to training images
        """

        # Store the datapath
        self.datapath = datapath
        self.im_shape = (None, None, 3)
        self.crop_im_shape = (256, 256, 3)
        self.total_imgs = None
        self.k = 1
        self.vgg = self.build_vgg()
        print('Initiating DataLoader with data from {}'.format(datapath))
        
        # Check data source
        if self.datapath.startswith('gs://'):
            self.content_bucket = os.path.join(self.datapath, 'content')
            self.style_bucket = os.path.join(self.datapath, 'style')
            print('Content bucket: ', self.content_bucket)
            print('Style bucket: ', self.style_bucket)
            self.content_img_paths = [os.path.join(self.content_bucket, i) for i in listdir(self.content_bucket)]
            self.style_img_paths = [os.path.join(self.style_bucket, i) for i in listdir(self.style_bucket)]
            self.num_content_pics = len(self.content_img_paths)
            self.num_style_pics = len(self.style_img_paths)

            print(">> Found {} content images in dataset".format(self.num_content_pics))
            print(">> Found {} style images in dataset".format(self.num_style_pics))
        else:
            self.style_img_paths = []
            self.content_img_paths = []
            for dirpath, _, filenames in os.walk(self.datapath):
                for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]:
                    if 'content' in dirpath:
                        self.content_img_paths.append(os.path.join(dirpath, filename))
                    elif 'style' in dirpath:
                        self.style_img_paths.append(os.path.join(dirpath, filename))

            self.num_content_pics = len(self.content_img_paths)
            self.num_style_pics = len(self.style_img_paths)

            print(">> Found {} content images in dataset".format(self.num_content_pics))
            print(">> Found {} style images in dataset".format(self.num_style_pics))


    def build_vgg(self):
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.im_shape)
        vgg.outputs = vgg.get_layer('block4_conv1').output
        model = Model(inputs=vgg.inputs, outputs=vgg.outputs)
        model.trainable = False
        return model
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]
    
    def load_img(self, path):
        if path.startswith('gs://'):
            path = file_io.FileIO(path, 'rb')
        img = image.load_img(path, target_size=(256,256))
        img = self.random_crop(np.array(img), random_crop_size=(self.crop_im_shape[0], self.crop_im_shape[1]))
        img = preprocess_input(np.array(img))
        return img

    def gen(self):
        while True:
            try:
                Fcs, Ic, Is = self.load_batch() 
                yield Fcs, Ic, Is
            except Exception as e:
                print(e)
                continue

    def __call__(self):
        while True:
            yield self.load_batch()

    def __next__(self):
        return self.load_batch()
    
    def __len__(self):
        return len(self.content_img_paths)
    
    def __getitem__(self, idx):
        return self.load_batch(idx=idx)      

    def load_batch(self, idx=0, img_paths=None, training=True):
        """Loads a batch of images from datapath folder"""     

        content_idx = np.random.randint(0, self.num_content_pics)
        content_img = self.load_img(self.content_img_paths[content_idx])
        vgg_content_img = np.expand_dims(content_img, 0)

        style_idx = np.random.randint(0, self.num_style_pics)
        style_img = self.load_img(self.style_img_paths[style_idx])
        vgg_style_img = np.expand_dims(style_img, 0)

        Fs = np.array(self.vgg(vgg_style_img))
        Fc = np.array(self.vgg(vgg_content_img))

        # alpha = np.random.uniform(low=1, high=1)
        # Fcs = get_Fcs(Fc, Fs, k=self.k, alpha=alpha)

        Fcs = Fc.reshape((32, 32, 512))

        return Fcs, content_img, style_img


def restore_original_image(x, data_format='channels_first'):

        mean = [103.939, 116.779, 123.68]
        # Zero-center by mean pixel
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] += mean[0]
                x[1, :, :] += mean[1]
                x[2, :, :] += mean[2]
            else:
                x[:, 0, :, :] += mean[0]
                x[:, 1, :, :] += mean[1]
                x[:, 2, :, :] += mean[2]
        else:
            x[..., 0] += mean[0]
            x[..., 1] += mean[1]
            x[..., 2] += mean[2]

        if data_format == 'channels_first':
            # 'BGR'->'RGB'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'BGR'->'RGB'
            x = x[..., ::-1]

        return x.astype(np.uint8)


def create_tf_dataset(datapath):
    output_types=(
        tf.float32,
        tf.float32,
        tf.float32
    )
    output_shapes=(
        tf.TensorShape([None, None, None]),
        tf.TensorShape([None, None, None]),
        tf.TensorShape([None, None, None])
    )

    ds = tf.data.Dataset.from_generator(DataLoader(datapath).gen,
                                    output_types=output_types, 
                                    output_shapes=output_shapes)

    return ds




def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def load_weights_from_gcs(gs_path_to_weights):
    # read the file from GCS
    weights_file = file_io.FileIO(gs_path_to_weights, mode='rb')
    # save model in temp dir
    temp_weights_location = './weights.h5'
    temp_weights_file = open(temp_weights_location, 'wb')
    temp_weights_file.write(weights_file.read())
    temp_weights_file.close()
    weights_file.close()

    return temp_weights_location



def plot_test_images(Ics, Ic, Is, log_test_path, epoch, filename='test_output'):

    Ic = np.array(Ic)
    Is = np.array(Is)
    Ics = np.array(Ics)

    Ocs = restore_original_image(Ics, 'channels_last')
    Oc = restore_original_image(Ic, 'channels_last')
    Os = restore_original_image(Is, 'channels_last')


    print(Ocs)
    print('Ocs ', np.where(Ocs == 255, 1, 0).sum())
    print('Ocs ', np.where(Ocs == 0, 1, 0).sum())

    print('Oc ', np.where(Oc == 255, 1, 0).sum())
    print('Os ', np.where(Os == 255, 1, 0).sum())

    # Images and titles
    images = {
        'Input': Oc, 
        'Style': Os, 
        'Output': Ocs
    }
                  
    fig, axes = plt.subplots(1, 3, figsize=(40, 10))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title("{} - {}".format(title, img.shape))
        axes[i].axis('off')

    plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
       
    file_name = "{}-Epoch_{}.png".format(filename, epoch)
    # Save the plot

    print('Saving the plot')
    if log_test_path.startswith('gs://'):
        fig.savefig(file_name)
        copy_file_to_gcs(job_dir=log_test_path, file_path=file_name)
    else:
        file_name = os.path.join(log_test_path, file_name)
        fig.savefig(file_name)

    plt.close()
    gc.collect()



