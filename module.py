from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def dis_z(z, options, reuse=False, name="dis_z"):

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        z = linear(z, options.gf_dim *32*32)
        z_ = tf.reshape(z, [-1, 32, 32, options.gf_dim])
        h0 = lrelu(conv2d(z_, options.df_dim, ks=5,s=2, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=5,s=2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (8 x 8 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4,ks=5,s=2, name='d_h2_conv'), 'd_bn2'))
        # h2 is (4x 4 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=5,s=2, name='d_h3_conv'), 'd_bn3'))
        # h3 is (2 x 2 x self.df_dim*8)
        h3=tf.reshape(h3, [-1, options.df_dim*8*4])

        h4 = linear(h3, 1, 'd_h4_lin')

        return h4

def encoder(image, options, reuse=False, name="encoder"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        s = int(options.image_size / 16)
        d1 = conv2d(image, options.gf_dim, 5, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = conv2d(d1, options.gf_dim*2, 5, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d3 = conv2d(d2, options.gf_dim*4, 5, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))
        d4 = conv2d(d3, options.gf_dim*8, 5, 2, name='g_d4_dc')
        d4 = tf.nn.relu(instance_norm(d4, 'g_d4_bn'))
        d4 = tf.reshape(d4, [-1,options.gf_dim *8*s*s])
        z1 = linear(d4, options.gf_dim * 4 * s * s, scope='l1')
        z1 = tf.nn.relu(batch_norm(z1, 'g_z_bn1'))
        z2 = linear(z1, options.gf_dim * 2 * s * s, scope='l2')
        z2 = tf.nn.relu(batch_norm(z2, 'g_z_bn2'))
        z3 = linear(z2, options.gf_dim * 1 * s * s, scope='l3')
        z3 = tf.nn.relu(batch_norm(z3, 'g_z_bn3'))
        z4 = linear(z3, 128)
        return z4

def generator(z, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        s = int(options.image_size/16)
        z1 = linear(z, options.gf_dim * 1 * s * s, scope='l1')
        z1 = tf.nn.relu(batch_norm(z1,'g_z_bn1'))
        z2 = linear(z1, options.gf_dim * 2 * s * s, scope='l2')
        z2 = tf.nn.relu(batch_norm(z2, 'g_z_bn2'))
        z3 = linear(z2, options.gf_dim * 4 * s * s, scope='l3')
        z3 = tf.nn.relu(batch_norm(z3, 'g_z_bn3'))
        z4=linear(z3,options.gf_dim*8*s*s, scope='l4')
        z_= tf.reshape(z4,[-1,s,s,options.gf_dim*8])
        z_ = tf.nn.relu(instance_norm(z_,'g_z_bn'))
        d1 = deconv2d(z_, options.gf_dim*4, 5, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim*2, 5, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d3 = deconv2d(d2, options.gf_dim*2, 5, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))
        d4 = deconv2d(d3, options.gf_dim, 5, 2, name='g_d4_dc')
        d4 = tf.nn.relu(instance_norm(d4, 'g_d4_bn'))
        d5 = tf.pad(d4, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        pred = conv2d(d5, options.output_c_dim, 3, 1, padding='VALID', name='g_pred_c')
        # pred = tf.nn.tanh(instance_norm(pred, 'g_pred_bn'))
        pred = tf.nn.tanh(pred)
        return pred

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))