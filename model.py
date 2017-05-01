from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64, br_initial=0.0,
                 anneal_rate=0.0002, gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 dataset_name='default', input_fname_pattern='*.jpg', architecture='dcgan', 
                 checkpoint_dir=None, sample_dir=None):
        """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.br_initial = br_initial
        self.anneal_rate = anneal_rate
        self.architecture = architecture

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, _ = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            self.c_dim = imread(self.data[0]).shape[-1]

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)
        self.bluffing_rate = tf.Variable(self.br_initial, name="bluffing_rate")

        self.num_bluffs_var = tf.placeholder(tf.float32, name="num_bluffs")
        self.bluffing_rate_var = tf.placeholder(tf.float32, name="bluffing_rate")
        self.bluff_summary_str = merge_summary([scalar_summary("num_bluffs", self.num_bluffs_var), scalar_summary("bluffing_rate", self.bluffing_rate_var)])

        if self.y_dim:
            n_bluffs = tf.to_int32(tf.multiply(self.bluffing_rate, self.batch_size))
            self.G = self.generator(self.z, self.y)
            if self.architecture == 'dcgan' or self.architecture == 'wgan':
                G_bluffed, self.G = tf.split(self.G, [n_bluffs, self.batch_size - n_bluffs])
            elif self.architecture == 'wgan-gp':
                G_bluffed, self.G = tf.split(self.G, [n_bluffs, self.batch_size + n_bluffs])
            inputs = tf.concat([inputs, G_bluffed], axis=0)
            self.D, self.D_logits = \
                self.discriminator(inputs, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = \
                self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            n_bluffs = tf.to_int32(tf.multiply(self.bluffing_rate, self.batch_size))
            G_bluffed, self.G = tf.split(self.G, [n_bluffs, self.batch_size - n_bluffs])
            inputs = tf.concat([inputs, G_bluffed], axis=0)
            self.D, self.D_logits = self.discriminator(inputs)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        if self.architecture == 'dcgan':
            self.d_loss_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
            self.g_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        elif self.architecture == 'wgan':
            self.d_loss_real = -tf.reduce_mean(self.D_logits)
            self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            self.g_loss = -tf.reduce_mean(self.D_logits_)
        elif self.architecture == 'wgan-gp':
            # TODO
            pass
            # self.d_loss_real = -tf.reduce_mean(self.D_logits)
            # self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            # self.g_loss = -tf.reduce_mean(self.D_logits_)

            """
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE, 1, 1],
                minval=0.,
                maxval=1.
            )
            fake_inputs, real_inputs = g, inputs
            differences = fake_inputs - real_inputs
            interpolates = real_inputs + (alpha*differences)
            print interpolates.get_shape()
            gradients = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
            print gradients.get_shape()
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            d_loss += LAMBDA*gradient_penalty
            """

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        if self.architecture == 'dcgan':
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
        elif self.architecture == 'wgan':
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
            gvs = d_optim.compute_gradients(self.d_loss, self.d_vars)
            clippable_count = tf.reduce_sum([tf.reduce_sum(tf.cast(tf.logical_and(tf.less_equal(grad, 0.01), tf.greater_equal(grad, -0.01)), tf.float32)) for grad, _ in gvs])
            clippable_count_sum = histogram_summary("clipped_count", clippable_count)
            capped_gvs = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in gvs]
            d_optim = d_optim.apply_gradients(capped_gvs)
        elif self.architecture == 'wgan-gp':
            # TODO
            pass
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        if self.architecture == 'dcgan':
            self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        elif self.architecture == 'wgan':
            self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, clippable_count_sum])
        elif self.architecture == 'wgan-gp':
            # TODO
            pass
        self.writer = SummaryWriter("./logs/{0}/arch={1}_bluff={2}_anneal={3}".format(self.dataset_name, self.architecture, self.br_initial, self.anneal_rate), self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            # sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          grayscale=self.grayscale) for sample_file in sample_files]
            if (self.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        prev_g_loss = curr_g_loss = 0.
        num_bluffs = 0.5 * config.batch_size
        bluffing_rate = self.br_initial
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob(os.path.join(
                    "./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    # batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                else:
                    batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch = [
                        get_image(batch_file,
                                  input_height=self.input_height,
                                  input_width=self.input_width,
                                  resize_height=self.output_height,
                                  resize_width=self.output_width,
                                  crop=self.crop,
                                  grayscale=self.grayscale) for batch_file in batch_files]
                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                if self.anneal_rate == 'auto':
                    if idx != 0 and idx % 1000 == 0:
                        if prev_g_loss > curr_g_loss:
                            num_bluffs = max(0, num_bluffs - 1)
                            bluffing_rate = num_bluffs * 1.0 / config.batch_size
                        elif prev_g_loss < curr_g_loss:
                            # TODO: This is sorta hardcoded
                            num_bluffs = min(0.5* config.batch_size, num_bluffs + 1)
                            bluffing_rate = num_bluffs * 1.0 / config.batch_size
                        print 'Updated num_bluffs, bluffing_rate: ', num_bluffs, bluffing_rate
                else:
                    bluffing_rate = self.br_initial * np.exp(-self.anneal_rate * counter)
                    num_bluffs = int(bluffing_rate*config.batch_size)

                if self.architecture == 'dcgan' or self.architecture == 'wgan':
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                        .astype(np.float32)
                elif self.architecture == 'wgan-gp':
                    # TODO
                    batch_z = np.random.uniform(-1, 1, [config.batch_size + 2*num_bluffs, self.z_dim]) \
                        .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z,
                                                   self.bluffing_rate: bluffing_rate
                                               })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.bluffing_rate: bluffing_rate
                                               })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.bluffing_rate: bluffing_rate})
                self.writer.add_summary(summary_str, counter)

                summary_str = self.sess.run(self.bluff_summary_str,
                                               feed_dict={self.bluffing_rate_var: bluffing_rate, self.num_bluffs_var: num_bluffs})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.z: batch_z
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if idx != 0 and idx % 1000 == 0:
                    prev_g_loss = curr_g_loss
                    curr_g_loss = errG

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            }
                        )
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

                # HACK: Hard-coded in based on input image sized used for each dataset instead of the more
                # complex route of using a batchsize tf variable (deriving it form the shape wouldn't work,
                # because of a tf bug that returns a vector of Nones for shapes after a split; though 
                # deriving it from self.input_height is an option, if slightly more complicated)
                if self.input_height == 28:
                    h4 = linear(tf.reshape(h3, [-1, 2048]), 1, 'd_h3_lin')
                elif self.input_height == 108:
                    h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                if self.architecture == 'dcgan' or self.architecture == 'wgan':
                    n_bluffs = 0
                elif self.architecture == 'wgan-gp':
                    n_bluffs = tf.to_int32(tf.multiply(self.bluffing_rate, self.batch_size))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size + 2*n_bluffs, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size + 2*n_bluffs, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size + 2*n_bluffs, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size + 2*n_bluffs, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size + 2*n_bluffs, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size + 2*n_bluffs, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size + 2*n_bluffs, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size + 2*n_bluffs, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
