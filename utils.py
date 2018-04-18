# create custom model class
import warnings

import numpy as np
import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import os
FLAGS = tf.app.flags.FLAGS


class Model(object):
    """A neural network model."""
    def __init__(self, name, inputs, params, mask):
        self.name = name
        self.mask = mask
        self.outputs = [inputs]
        self.layer_num = 0
        # default mode is initialization
        self.init = True
        self.params = OrderedDict()

        if params is not None:
            self.params.update(params)
            self.init = False

    def _get_layer_str(self, layer=None):
        """function for setting layers names"""
        if layer is None:
            layer = self.layer_num

        return '%s_h%d' % (self.name, layer)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    @staticmethod
    def _glorot_initializer(prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev = np.sqrt(stddev_factor / np.sqrt(prev_units * num_units))
        weights = tf.truncated_normal([prev_units, num_units], mean=0.0, stddev=stddev)
        return weights

    @staticmethod
    def phase_shift(inputs, scale, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)

    @staticmethod
    def _glorot_initializer_conv2d(prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev = np.sqrt(stddev_factor / (np.sqrt(prev_units * num_units) * mapsize * mapsize))
        weights = tf.truncated_normal([mapsize, mapsize, prev_units, num_units], mean=0.0, stddev=stddev)

        return weights

    def _write_params(self, weight, bias, layer_name):
        # write model params
        self.params[layer_name + '_W'] = weight
        self.params[layer_name + '_b'] = bias
        self.layer_num += 1

    def _get_params(self, layer_name):
        try:
            W = self.params[layer_name + '_W']
            b = self.params[layer_name + '_b']
        except KeyError:
            raise LookupError('No params for ' + layer_name + 'specified')
        self.layer_num += 1
        return W, b

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str() + '_batch_norm' + self.mask, reuse=tf.AUTO_REUSE):
            out = tf.layers.batch_normalization(self.get_output(), scale=scale)

        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str() + self.mask):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""

        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        layer_name = self._get_layer_str() + '_lin'
        if self.init:
            with tf.variable_scope(layer_name + self.mask, reuse=tf.AUTO_REUSE):
                prev_units = self._get_num_inputs()

                # Weight term
                initw = self._glorot_initializer(prev_units, num_units, stddev_factor=stddev_factor)
                weight = tf.get_variable('weight', initializer=initw)
                # Bias term
                initb = tf.constant(0.0, shape=[num_units])
                bias = tf.get_variable('bias', initializer=initb)

                self._write_params(weight, bias, layer_name)
        else:
            weight, bias = self._get_params(layer_name)

        out = tf.matmul(self.get_output(), weight) + bias
        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        out = tf.nn.sigmoid(self.get_output())
        self.outputs.append(out)
        return self

    def add_tanh(self):
        """Adds a tanh (-1, 1) activation function layer to this model"""
        out = tf.nn.tanh(self.get_output())
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""


        this_input = tf.square(self.get_output())
        reduction_indices = list(range(1, len(this_input.get_shape())))
        acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
        out = this_input / (acc + FLAGS.epsilon)

        self.outputs.append(out)
        return self

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str() + self.mask, reuse=tf.AUTO_REUSE):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str() + self.mask, reuse=tf.AUTO_REUSE):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str() + self.mask, reuse=tf.AUTO_REUSE):
            t1 = .5 * (1 + leak)
            t2 = .5 * (1 - leak)
            out = t1 * self.get_output() + t2 * tf.abs(self.get_output())

        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0, padding='SAME'):
        """Adds a 2D convolutional layer."""

        assert len(
            self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        layer_name = self._get_layer_str() + '_conv'
        if self.init:
            with tf.variable_scope(layer_name + self.mask, reuse=tf.AUTO_REUSE):
                prev_units = self._get_num_inputs()

                # Weight term and convolution
                initw = self._glorot_initializer_conv2d(prev_units, num_units, mapsize, stddev_factor=stddev_factor)
                weight = tf.get_variable('weight', initializer=initw)
                # out = tf.nn.conv2d(self.get_output(), weight, strides=[1, stride, stride, 1], padding=padding)
                # Bias term
                initb = tf.constant(0.0, shape=[num_units])
                bias = tf.get_variable('bias', initializer=initb)

                self._write_params(weight, bias, layer_name)

        else:
            weight, bias = self._get_params(layer_name)

        out = tf.nn.conv2d(self.get_output(), weight, strides=[1, stride, stride, 1], padding=padding)
        out = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        layer_name = self._get_layer_str() + '_tconv'
        prev_output = self.get_output()
        output_shape = [FLAGS.batch_size,
                        int(prev_output.get_shape()[1]) * stride,
                        int(prev_output.get_shape()[2]) * stride,
                        num_units]
        if self.init:
            with tf.variable_scope(layer_name + self.mask, reuse=tf.AUTO_REUSE):
                prev_units = self._get_num_inputs()

                # Weight term and convolution
                initw = self._glorot_initializer_conv2d(prev_units, num_units, mapsize, stddev_factor=stddev_factor)

                weight = tf.get_variable('weight', initializer=initw)
                weight = tf.transpose(weight, perm=[0, 1, 3, 2])

                # out = tf.nn.conv2d_transpose(self.get_output(), weight,
                #                             output_shape=output_shape,
                #                             strides=[1, stride, stride, 1],
                #                             padding='SAME')

                # Bias term
                initb = tf.constant(0.0, shape=[num_units])
                bias = tf.get_variable('bias', initializer=initb)

                self._write_params(weight, bias, layer_name)
        else:
            weight, bias = self._get_params(layer_name)

        out = tf.nn.conv2d_transpose(self.get_output(), weight,
                                     output_shape=output_shape,
                                     strides=[1, stride, stride, 1],
                                     padding='SAME')

        out = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            # bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units // 4, mapsize=1, stride=1, stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units // 4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units // 4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=2.)

        self.add_sum(bypass)

        return self

    # The implementation of PixelShuffler
    def add_pixel_shuffler(self, scale=2):
        inputs = self.get_output()
        size = tf.shape(inputs)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = inputs.get_shape().as_list()[-1]

        # Get the target channel size
        channel_target = c / (scale * scale)
        channel_factor = c / channel_target

        shape_1 = [batch_size, h, w, channel_factor / scale, channel_factor / scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        # Reshape and transpose for periodic shuffling for each channel
        input_split = tf.split(inputs, channel_target, axis=3)
        output = tf.concat([self.phase_shift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
        # Reshape using shape from previous output to avoid uncertainty
        new_shape = inputs.get_shape().as_list()
        new_shape[1] *= 2
        new_shape[2] *= 2
        new_shape[3] /= (scale * scale)
        self.outputs.append(tf.reshape(output, new_shape))
        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str() + self.mask, reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            # print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)

        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str() + self.mask, reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)

        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def summary(self):
        for key in self.params.keys():
            print key, self.params[key].shape


def vgg_19(inputs, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True,
           scope='vgg_19', reuse=False, fc_conv_padding='VALID'):
    """Oxford Net VGG 19-Layers version E Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
    Returns:
    the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net, end_points


def VGG19_slim(input, type, reuse, scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer = 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output


def downscale(images, K):

    warnings.warn("this function will be removed in future.", FutureWarning)

    arr = np.zeros([K, K, 3, 3])
    arr[:, :, 0, 0] = 1.0 / (K * K)
    arr[:, :, 1, 1] = 1.0 / (K * K)
    arr[:, :, 2, 2] = 1.0 / (K * K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)

    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled


def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr


def huber_loss(labels, predictions, delta=1.0):

    warnings.warn("this function will be removed in future.", FutureWarning)

    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def print_images(sampled_images, label, index, directory, save_all_samples=False):
    # mpl.use('Agg')  # for server side

    def unnormalize(img, cdim):
        img_out = np.zeros_like(img)
        for i in xrange(cdim):
            img_out[:, :, i] = 255. * ((img[:, :, i] + 1.) / 2.0)
        img_out = img_out.astype(np.uint8)
        return img_out

    if type(sampled_images) == np.ndarray:
        N, h, w, cdim = sampled_images.shape
        idxs = np.random.choice(np.arange(N), size=(4, 4), replace=False)
    else:
        sampled_images, idxs = sampled_images
        N, h, w, cdim = sampled_images.shape

    fig, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            if cdim == 1:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim)[:, :, 0], cmap="gray")
            else:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim))
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')

    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, "%s_%i.png" % (label, index)), bbox_inches='tight')
    plt.close("all")

    if "raw" not in label.lower() and save_all_samples:
        np.savez_compressed(os.path.join(directory, "samples_%s_%i.npz" % (label, index)),
                            samples=sampled_images)


def load_weights(sess, mode=-1):
    # load last saved weights (this function needs simplification. maybe it is possible to load wghts like in setup_vgg)

    wpaths = filter(lambda x: x.startswith('weights'), os.listdir(FLAGS.checkpoint_dir))
    wpaths = sorted(wpaths, key=lambda x: int(x[x.rfind('_') + 1 : x.rfind('.')]))
    if len(wpaths) == 0:
        raise RuntimeError('Can\'t find any weights in checkpoint directory')

    if mode == -1:
        path = wpaths[-1]  # take the last
    else:
        path = 'weights_%i.npz' % mode

    weights = np.load(FLAGS.checkpoint_dir + '/' + path)
    start_it = int(path[path.rfind('_') + 1: path.rfind('.')])

    print 'Weights from %i' % start_it, 'iteration loaded!'
    # assign loaded to existing
    for var_name, var_arr in weights.iteritems():
        # is there a way to simplify this?
        sess.run(tf.assign(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_name)[0], var_arr))

    print 'Model weights restored successfully!'
    return start_it + 1


def setup_vgg(sess):
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)
    try:
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)
    except:
        raise LookupError("seems like you don't have vgg weights or checkpoint path (vgg_cktp) is wrong")
    print 'VGG19 restored successfully!'
