import os
import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
from trainer.MST import MST
from trainer.utils import create_tf_dataset, plot_test_images
from datetime import datetime


def train_mst(
    datapath,
    num_epochs,
    steps_per_epoch,
    log_tensorboard_path,
    log_weight_path,
    log_test_path,
    batch_size=1,
    decoder_weights=None,
    **kwargs
    ):

    # create model
    model = MST(256, 256, 3, decoder_weights=decoder_weights)

    # create data input
    ds = create_tf_dataset(datapath=datapath)
    ds = ds.map(map_func=lambda Fcs,Ic,Is: (Fcs, Ic, Is), num_parallel_calls=32).batch(batch_size).prefetch(1)

    # create optimizera
    opt = tf.optimizers.Adam(1e-4)

    # tensorboard summary
    summary_writer = tf.summary.create_file_writer(log_tensorboard_path)

    # metrics
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    content_loss_metric = tf.keras.metrics.Mean(name='train_content_loss')
    style_loss_metric = tf.keras.metrics.Mean(name='train_style_loss')

    @tf.function
    def train_step(Fcs, Ic, Is):

        with tf.GradientTape() as tape:
            Ics = model.decoder(Fcs)
            content_loss, style_loss = model.get_loss(Ics, Ic, Is, batch_size=batch_size)
            loss = 0.01 * style_loss + content_loss
            gradients = tape.gradient(loss, model.decoder.trainable_variables)
            gvs = zip(gradients, model.decoder.trainable_variables)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            opt.apply_gradients(gvs)

        # Update the metrics
        loss_metric.update_state(loss)
        content_loss_metric.update_state(content_loss)
        style_loss_metric.update_state(style_loss)
        return Ics


    for epoch in range(num_epochs):
        # Reset the metrics
        loss_metric.reset_states()
        style_loss_metric.reset_states()
        content_loss_metric.reset_states()

        for Fcs, Ic, Is in ds.take(steps_per_epoch):
            Ics = train_step(Fcs, Ic, Is)


        # Tensorboard
        mean_loss = loss_metric.result()
        mean_content_loss = content_loss_metric.result()
        mean_style_loss = style_loss_metric.result()

        with summary_writer.as_default():
            tf.summary.scalar('loss', mean_loss, step=epoch)
            tf.summary.scalar('content_loss', mean_content_loss, step=epoch)
            tf.summary.scalar('style_loss', mean_style_loss, step=epoch)

        # Save Weights
        model.save_decoder_weights(log_weight_path, epoch=epoch)
    
        # Print output
        print('Testing images')
        plot_test_images(Ics[0, ...], Ic[0, ...], Is[0, ...], log_test_path, epoch)

        # Print Logs
        print('Epoch: ', epoch)
        print('Loss: {:.3f}'.format(mean_loss / 100000))
        print('Content Loss: {:.3f}'.format(mean_content_loss / 100000))
        print('Style Loss: {:.3f}'.format(mean_style_loss / 100000))

   

def parse_args():
    parser = ArgumentParser(description='Training script for MST')

    parser.add_argument(
        '-datapath', '--datapath',
        type=str, default='./data/',
        help='Folder with training images'
    )
     
    parser.add_argument(
        '-job-dir', '--job-dir',
        type=str, default='./jobs/1/',
        help='Job Directory'
    )

    parser.add_argument(
        '-weights', '--weights',
        type=str,
        required=False,
        default=None,
        help='Path to decoder weights'
    )

        
    return  parser.parse_args()



# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
       
    # Common settings for all training stages
    common = {
        "job_dir": args.job_dir,
        "batch_size": 1, 
        "datapath": args.datapath,
        "num_epochs":1000,
        "steps_per_epoch": 5,
        "decoder_weights": args.weights,
        "log_weight_path": os.path.join(args.job_dir, 'weights'),
        "log_tensorboard_path": os.path.join(args.job_dir, 'tensorboard'),        
        "log_test_path": os.path.join(args.job_dir, 'test_output'),
    }

    if not common['job_dir'].startswith('gs://'):
        if not os.path.exists(common['job_dir']):
            os.makedirs(common['job_dir'])
        if not os.path.exists(common['log_weight_path']):
            os.makedirs(common['log_weight_path'])
        if not os.path.exists(common['log_tensorboard_path']):
            os.makedirs(common['log_tensorboard_path'])
        if not os.path.exists(common['log_test_path']):
            os.makedirs(common['log_test_path'])

    train_mst(**common)

        






