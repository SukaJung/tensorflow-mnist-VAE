import tensorflow as tf
import numpy as np
import dataset
import os
import vae
import plot_utils
import glob

IMAGE_SIZE = 64

def main():
    dim_z = 100
    output_size = 64*64
    """ prepare MNIST data """

    train_total_data, train_size, _, _, test_data, test_labels = dataset.prepare_MNIST_data()
    n_samples = train_size

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, output_size], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, output_size], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, output_size, dim_z, output_size, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)

    """ training """
    # train
    total_batch = int(n_samples / 64)
    min_tot_loss = 1e99
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(100000):
            batch_xs_input = dataset.getbatch(64)
            batch_xs_target = batch_xs_input
            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9})

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss,loss_likelihood, loss_divergence))





if __name__ == '__main__':
    main()