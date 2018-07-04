from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# CUDA_VISIVLE_DEVICES=1 python main.py
import h5py
from module import *
from utils import *


class mixgan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.z_size = args.z_dim
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.dis_z = dis_z
        self.encoder =encoder
        self.generator = generator_a
        self.discriminator = discriminator
        self.decoder = decoder
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim input_c_dim z_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc, args.input_nc, args.z_dim))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.real_A = tf.placeholder(tf.float32,
                                      [None, self.image_size, self.image_size,
                                       self.output_c_dim],
                                      name='real_images_A')
        self.real_B = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name='real_images_B')

        self.z = tf.placeholder(tf.float32,[None, self.z_size], name='noise_z')
        self.fake_z = self.encoder(self.real_B, self.options, False, name="encoder")
        self.fake_BB, dpb = self.decoder(self.fake_z, self.options, False, name="decoder")
        self.test_BB, dpz = self.decoder(self.z, self.options, True, name="decoder")

        self.fake_A = self.generator(dpz, self.options, False, name="generator")

        self.DA_fake = self.discriminator(self.fake_A, self.options, False, name="dis_A")

        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))
        self.DA_real = self.discriminator(self.real_A, self.options, True, name="dis_A")
        self.dA_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.dA_loss_fake = self.criterionGAN(self.DA_fake, tf.zeros_like(self.DA_fake))
        self.dA_loss = (self.dA_loss_real + self.dA_loss_fake) / 2

        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.dA_loss_sum = tf.summary.scalar("dA_loss", self.dA_loss)
        self.dA_loss_real_sum = tf.summary.scalar("dA_loss_real", self.dA_loss_real)
        self.dA_loss_fake_sum = tf.summary.scalar("dA_loss_fake", self.dA_loss_fake)
        self.dA_sum = tf.summary.merge(
            [self.dA_loss_sum, self.dA_loss_real_sum, self.dA_loss_fake_sum]
        )

        self.Dz_fake = self.dis_z(self.fake_z, self.options, reuse=False, name="dis_z")
        self.en_asg_loss = self.criterionGAN(self.Dz_fake, tf.ones_like(self.Dz_fake))
        self.x_ae_loss = self.L1_lambda * abs_criterion(self.fake_BB, self.real_B)
        self.en_loss = self.en_asg_loss + self.x_ae_loss
        self.de_g_loss = self.x_ae_loss + self.g_loss

        # + abs_criterion(self.fake_x, self.real_x)

        self.Dz_real = self.dis_z(self.z, self.options, reuse=True, name="dis_z")
        self.dz_loss_real = self.criterionGAN(self.Dz_real, tf.ones_like(self.Dz_real))
        self.dz_loss_fake = self.criterionGAN(self.Dz_fake, tf.zeros_like(self.Dz_fake))
        self.dz_loss = (self.dz_loss_real + self.dz_loss_fake) / 2

        self.en_sum = tf.summary.scalar("en_loss", self.en_loss)
        self.de_g_sum = tf.summary.scalar("de_g_loss", self.de_g_loss)
        self.x_ae_sum = tf.summary.scalar("x_ae_loss", self.x_ae_loss)
        self.en_asg_sum = tf.summary.scalar("en_asg_loss", self.en_asg_loss)
        self.de_g_summary = tf.summary.merge(
            [self.de_g_sum, self.x_ae_sum, self.g_sum]
        )
        self.en_summary = tf.summary.merge(
            [self.en_sum, self.x_ae_sum, self.en_asg_sum]
        )

        self.dz_loss_sum = tf.summary.scalar("dz_loss", self.dz_loss)
        self.dz_loss_real_sum = tf.summary.scalar("dz_loss_real", self.dz_loss_real)
        self.dz_loss_fake_sum = tf.summary.scalar("dz_loss_fake", self.dz_loss_fake)
        self.dz_sum = tf.summary.merge(
            [self.dz_loss_sum, self.dz_loss_real_sum, self.dz_loss_fake_sum]
        )

        t_vars = tf.trainable_variables()
        self.dz_vars = [var for var in t_vars if 'dis_z' in var.name]
        self.de_vars = [var for var in t_vars if 'decoder' in var.name]
        self.en_vars = [var for var in t_vars if 'encoder' in var.name]
        self.dA_vars = [var for var in t_vars if 'dis_A' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train mixgan"""
        self.dz_optim = tf.train.AdamOptimizer(args.lr/50, beta1=args.beta1) \
            .minimize(self.dz_loss, var_list=self.dz_vars)
        self.en_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.en_loss, var_list=self.en_vars)
        self.dA_optim = tf.train.AdamOptimizer(args.lr/10, beta1=args.beta1) \
            .minimize(self.dA_loss, var_list=self.dA_vars) #1/2
        self.de_g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)#1/1
            # .minimize(self.de_g_loss, var_list=[self.g_vars,self.de_vars])


            # .minimize(self.de_g_loss, var_list=self.de_vars)




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
            dataA = h5py.File('/home/guangyuan/Downloads/handbag_64.hdf5', 'r')['imgs']
            dataB = h5py.File('/home/guangyuan/Downloads/shoes_64.hdf5', 'r')['imgs']

            batch_idxs = min(len(dataA), len(dataB), args.train_size) // self.batch_size
            for idx in range(0, batch_idxs):
                batch_filesA = dataA[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_imagesA = load_data(batch_filesA).astype(np.float32)
                batch_filesB = dataB[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_imagesB = load_data(batch_filesB).astype(np.float32)
                batch_imagesB1, batch_imagesB2, batch_imagesB3 = np.split(batch_imagesB, 3, axis=3)
                batch_imagesB = batch_imagesB1 * 0.114 + batch_imagesB2 * 0.587 + batch_imagesB3 * 0.299
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size]) \
                    .astype(np.float32)
                if epoch<100:
                    # Update D network
                    _, summary_str = self.sess.run([self.dz_optim, self.dz_sum],
                                                   feed_dict={self.real_B: batch_imagesB,
                                                              self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([self.en_optim, self.en_summary],
                                                   feed_dict={self.real_B: batch_imagesB,
                                                              self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)
                if epoch > 99:
                    # Update D network
                    _, summary_str = self.sess.run([self.dA_optim, self.dA_sum],
                                                   feed_dict={self.real_A: batch_imagesA,
                                                              self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([self.de_g_optim, self.de_g_summary],
                                                   feed_dict={self.real_A: batch_imagesA,
                                                              self.real_B: batch_imagesB,
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
        model_name = "mixgan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):


        # dataA = h5py.File('/home/guangyuan/Downloads/handbag_64.hdf5', 'r')['imgs']
        dataB = h5py.File('/home/guangyuan/Downloads/shoes_64.hdf5', 'r')['imgs']


        # batch_filesA = dataA[idx * self.batch_size: (idx + 1) * self.batch_size]
        # batch_imagesA = load_data(batch_filesA).astype(np.float32)
        batch_filesB = dataB[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imagesB = load_data(batch_filesB).astype(np.float32)
        batch_imagesB1, batch_imagesB2, batch_imagesB3 = np.split(batch_imagesB, 3, axis=3)
        batch_imagesB = batch_imagesB1 * 0.114 + batch_imagesB2 * 0.587 + batch_imagesB3 * 0.299
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size]) \
            .astype(np.float32)
        [fake_A,fake_BB,test_BB] = self.sess.run([self.fake_A,self.fake_BB,self.test_BB],
                               feed_dict={self.real_B: batch_imagesB,
                                          self.z: batch_z})

        save_images(fake_A, [int(self.batch_size/8), 8],
                    './{}/zA_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(test_BB, [int(self.batch_size/8), 8],
                    './{}/zBB_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_BB, [int(self.batch_size/8), 8],
                    './{}/rBB_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for k in range(100):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size]) \
                .astype(np.float32)
            [fake_A] = self.sess.run([self.fake_A],
                                     feed_dict={self.z: batch_z})
            save_images(fake_A, [int(np.sqrt(self.batch_size)), int(np.sqrt(self.batch_size))],
                        './{}/test_G_{:2d}.jpg'.format(args.test_dir, k))


