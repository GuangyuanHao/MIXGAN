from __future__ import division
import tensorflow as tf
from ops import *

def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 64 x 64 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim,ks=4,s=1, name='d_h0_conv'))
        # h0 is (64 x 64 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=2,s=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (32 x 32 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4,ks=1, s=1,name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h4 = conv2d(h2, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4
def dis_z(z, options, reuse=False, name="dis_z"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        z = linear(z, options.gf_dim *32*32)
        z_ = tf.reshape(z, [-1, 32, 32, options.gf_dim])
        h0 = lrelu(conv2d(z_, options.df_dim, ks=4,s=2, name='d_h0_conv'))
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=4,s=2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4,ks=4,s=2, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=4,s=2, name='d_h3_conv'), 'd_bn3'))
        h3=tf.reshape(h3, [-1, options.df_dim*8*4])
        h4 = linear(h3, 1, 'd_h4_lin')
        return h4


def encoder(image, options, reuse=False, name="encoder"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        # print("encoder",image.get_shape())
        image_size = int(options.image_size)
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim * 4, 3, 2, name='g_e4_c'), 'g_e4_bn'))
        c5 = tf.nn.relu(instance_norm(conv2d(c4, options.gf_dim * 8, 3, 2, name='g_e5_c'), 'g_e5_bn'))
        d4 = tf.reshape(c5, [-1,options.gf_dim *8*int((image_size/16)**2)])
        z1 = linear(d4, options.gf_dim * 4 * ((image_size/16)**2), scope='l1')
        z1 = tf.nn.relu(batch_norm(z1, 'g_z_bn1'))
        z2 = linear(z1, options.gf_dim * 4 * ((image_size/16)**2), scope='l2')
        z2 = tf.nn.relu(batch_norm(z2, 'g_z_bn2'))
        z3 = linear(z2, options.gf_dim * 2 * ((image_size/16)**2), scope='l3')
        z3 = tf.nn.relu(batch_norm(z3, 'g_z_bn3'))
        z4 = linear(z3, options.z_dim)
        return z4

def decoder(z, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        image_size = int(options.image_size)
        z1 = linear(z, options.gf_dim * 2 * ((image_size / 16) ** 2), scope='l1')
        z1 = tf.nn.relu(batch_norm(z1, 'g_z_bn1'))
        z2 = linear(z1, options.gf_dim * 4 * ((image_size / 16) ** 2), scope='l2')
        z2 = tf.nn.relu(batch_norm(z2, 'g_z_bn2'))
        z3 = linear(z2, options.gf_dim * 4 * ((image_size / 16) ** 2), scope='l3')
        z3 = tf.nn.relu(batch_norm(z3, 'g_z_bn3'))
        z4 = linear(z3, options.gf_dim * 8 * ((image_size / 16) ** 2), scope='l4')
        c0 = tf.reshape(z4, [-1, int(image_size / 16), int(image_size / 16), options.gf_dim * 8])
        c0 = tf.nn.relu(instance_norm(c0, 'g_z_bn'))
        c1 = deconv2d(c0, options.gf_dim * 8, 3, 2, name='g_c1_dc')  # 16
        c1 = tf.nn.relu(instance_norm(c1, 'g_c1_bn'))
        c2 = deconv2d(c1, options.gf_dim * 4, 3, 2, name='g_c2_dc')  # 32
        c2 = tf.nn.relu(instance_norm(c2, 'g_c2_bn'))
        # define G network with 6 resnet blocks
        r1 = residule_block(c2, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        d1 = deconv2d(r6, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim*1, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2_ = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2_, options.input_c_dim, 7, 1, padding='VALID', name='g_pred_c')
        pred = tf.nn.tanh(pred, 'g_pred_bn')
        return pred, tf.concat([d2,pred],axis=3)
def generator_a(dp, options, reuse=False, name="generator_y"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        d1 = deconv2d(dp, options.gf_dim * 2, 3, 1, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = tf.concat([d1,dp],axis=3)
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c')
        pred = tf.nn.tanh(pred, 'g_pred_bn')
        return pred

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))