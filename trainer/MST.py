import tensorflow as tf 
import numpy as np 
import os

from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Lambda, Conv2DTranspose, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from trainer.utils import copy_file_to_gcs, load_weights_from_gcs



class MST:

    def __init__(self, im_h, im_w, im_c, decoder_weights=None):
        self.im_shape = (im_h, im_w, im_c)
        self.decoder_weights = decoder_weights
        self.vgg_features_shape = (None, None, 512)
        self.vgg_loss_model = self.build_vgg_loss()
        self.kernel_size = (3,3)
        self.decoder = self.build_decoder()

        # load weights
        if self.decoder_weights is not None:
            if self.decoder_weights.startswith('gs://'):
                # write a local temp file to read weights from
                print('Setting the generator weights to {}'.format(self.decoder_weights))
                self.decoder_weights = load_weights_from_gcs(self.decoder_weights)
            # load the file
            self.decoder.load_weights(self.decoder_weights)
            print('Done setting the weights')
    

    def build_vgg_loss(self):
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.im_shape)
        vgg.outputs = [
            vgg.get_layer('block1_conv1').output,
            vgg.get_layer('block2_conv1').output,
            vgg.get_layer('block3_conv1').output,
            vgg.get_layer('block4_conv1').output
        ]

        model = Model(inputs=vgg.inputs, outputs=vgg.outputs)
        model.trainable = False

        return model


    def save_decoder_weights(self, weight_path, epoch):
        """Save the decoder network weights"""

        name = "decoder_{}.h5".format(epoch)

        if weight_path.startswith('gs://'):
            self.decoder.save_weights(name)
            copy_file_to_gcs(job_dir=weight_path, file_path=name)
        else:
            name = os.path.join(weight_path, name)
            self.decoder.save_weights(name)


    def build_decoder(self):
        """
        Mirrors the VGG network with max-pooling layers replaces by UpScaling Layers
        """

        Fcs = Input((None, None, 512))
        x = Conv2DTranspose(filters=256, kernel_size=self.kernel_size, padding='same', bias_initializer='zeros', activation='relu', kernel_initializer='glorot_uniform')(Fcs)

        x = UpSampling2D()(x)
        for _ in range(3):
            x = Conv2DTranspose(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu', bias_initializer='zeros')(x)
        
        x = UpSampling2D()(x)
        x = Conv2DTranspose(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu', bias_initializer='zeros')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=3, kernel_size=self.kernel_size, padding='same', bias_initializer='zeros')(x)

        model = Model(inputs=Fcs, outputs=x)

        return model


    def get_loss(self, Ics, Ic, Is, batch_size=2, epsilon=1e-6):

        vgg_model = self.vgg_loss_model

        def mse(x,y):
            '''Mean Squared Error'''
            return tf.reduce_mean(tf.math.squared_difference(x, y))

        def sse(x,y):
            '''Sum of Squared Error'''
            return tf.reduce_sum(tf.math.squared_difference(x, y))

        def content_loss(y_pred, y_true):
            current, target = vgg_model(y_pred)[3], vgg_model(y_true)[3]
            diff = current - target
            sq = K.square(diff)
            loss = K.mean(sq)
            # loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true)) # MSE 

            return loss

        def style_loss(y_pred, y_true):
            style_loss = 0.
        
            for _ in range(4):
                d_map, s_map = vgg_model(y_pred)[_], vgg_model(y_true)[_]

                s_mean, s_var = tf.nn.moments(s_map, axes=[1,2])
                d_mean, d_var = tf.nn.moments(d_map, axes=[1,2])

                d_std = tf.sqrt(d_var + epsilon)
                s_std = tf.sqrt(s_var + epsilon)

                mu_loss = sse(d_mean, s_mean) / batch_size
                std_loss = sse(d_std, s_std) / batch_size

                mu_std_loss = mu_loss + std_loss
                style_loss += mu_std_loss
            
            return style_loss


        def full_loss(Ics, Ic, Is):

            return content_loss(Ics, Ic), style_loss(Ics, Is)

        return full_loss(Ics, Ic, Is)
        