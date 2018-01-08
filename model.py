from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from module import *
from utils import *


class aae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.z_size = 128
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_name = args.dataset_name
        self.dis_z = dis_z
        self.encoder =encoder
        self.generator = generator
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.real_x = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name='real_images')
        self.z = tf.placeholder(tf.float32,[None, self.z_size], name='noise_z')

        self.fake_z = self.encoder(self.real_x, self.options, False, name="encoder")
        self.fake_x = self.generator(self.fake_z, self.options, False, name="generator")

        self.test_x = self.generator(self.z, self.options, True, name="generator")

        self.Dz_fake = self.dis_z(self.fake_z, self.options, reuse=False, name="dis_z")
        self.g_loss_z = self.criterionGAN(self.Dz_fake, tf.ones_like(self.Dz_fake))+ \
                        self.L1_lambda*abs_criterion(self.fake_x, self.real_x)

        self.Dz_real = self.dis_z(self.z, self.options, reuse=True, name="dis_z")
        self.dz_loss_real = self.criterionGAN(self.Dz_real, tf.ones_like(self.Dz_real))
        self.dz_loss_fake = self.criterionGAN(self.Dz_fake, tf.zeros_like(self.Dz_fake))
        self.dz_loss = (self.dz_loss_real + self.dz_loss_fake) / 2

        self.g_z_sum = tf.summary.scalar("g_loss_z", self.g_loss_z)

        self.dz_loss_sum = tf.summary.scalar("dz_loss", self.dz_loss)
        self.dz_loss_real_sum = tf.summary.scalar("dz_loss_real", self.dz_loss_real)
        self.dz_loss_fake_sum = tf.summary.scalar("dz_loss_fake", self.dz_loss_fake)
        self.dz_sum = tf.summary.merge(
            [self.dz_loss_sum, self.dz_loss_real_sum, self.dz_loss_fake_sum]
        )

        t_vars = tf.trainable_variables()
        self.dz_vars = [var for var in t_vars if 'dis_z' in var.name]
        self.gz_vars = [var for var in t_vars if 'generator' in var.name]
        self.en_vars = [var for var in t_vars if 'encoder' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):

        self.dz_optim = tf.train.AdamOptimizer(args.lr/100, beta1=args.beta1) \
            .minimize(self.dz_loss, var_list=self.dz_vars)
        self.gz_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_z, var_list=[self.gz_vars,self.en_vars])

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dir_path = '/home/guangyuan/CelebA/Img/img_align_celeba'
            names = os.listdir(dir_path)
            data = []
            for name in names:
                sr_path = dir_path + '/' + name
                data.append(sr_path)
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = list(data[idx * self.batch_size:(idx + 1) * self.batch_size])
                batch_images = [load_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_size]) \
                    .astype(np.float32)
                # Update D network
                _, summary_str = self.sess.run([self.dz_optim, self.dz_sum],
                                               feed_dict={self.real_x: batch_images,
                                                          self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update Encoder and Generator network
                _, summary_str = self.sess.run([self.gz_optim, self.g_z_sum],
                                               feed_dict={self.real_x: batch_images,
                                                          self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([self.gz_optim, self.g_z_sum],
                                               feed_dict={self.real_x: batch_images,
                                                          self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)


                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "aae.model"
        model_dir = "%s_%s" % (self.dataset_name, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dir_path = '/home/guangyuan/CelebA/Img/img_align_celeba'
        names = os.listdir(dir_path)
        data = []
        for name in names:
            sr_path = dir_path + '/' + name
            data.append(sr_path)
        np.random.shuffle(data)
        batch_files = list(data[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_images = [load_data(batch_file) for batch_file in batch_files]
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.normal(0, 1, [self.batch_size, self.z_size]) \
            .astype(np.float32)
        [fake_x,test_x,real_x] = self.sess.run([self.fake_x,self.test_x,self.real_x],
                               feed_dict={self.real_x: batch_images,
                                          self.z: batch_z})
        save_images(fake_x, [4, 4],
                    './{}/fake_x_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(test_x, [4, 4],
                    './{}/test_x_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(real_x, [4, 4],
                    './{}/real_x_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")



        for k in range(100):
            batch_z = np.random.normal(0, 1, [self.batch_size, self.z_size]) \
                .astype(np.float32)
            test_x = self.sess.run(self.test_x, feed_dict={self.z: batch_z})
            save_images(test_x, [int(np.sqrt(self.batch_size)), int(np.sqrt(self.batch_size))],
                        './{}/test_G_{:2d}.jpg'.format(args.test_dir, k))


