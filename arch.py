from utils import *

FLAGS = tf.app.flags.FLAGS


class BSRGAN(object):
    def __init__(self, hr_images, lr_images, dataset_size, batch_size=64,
                 prior_std=1.0, J=1, M=1, eta=2e-4,
                 alpha=0.01, lrate=0.0002, optimizer='adam',
                 ml=False, J_d=None):

        # J_d and and J are amount of theta_d and theta_g samples (see 2.1, marginalizing the noise)
        self.optimizer = optimizer.lower()
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.hr = hr_images
        self.lr = lr_images
        self.hr_dim = list(hr_images.shape[1:])
        # TODO: fix this crutch below
        self.lr_dim = [32, 32, 3]#list(lr_images.shape[1:])
        self.c_dim = self.hr_dim[2]  # channels
        self.lrate = lrate

        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_disc = J_d if J_d is not None else 1
        self.num_mcmc = M
        self.eta = eta
        self.alpha = alpha
        # ML
        self.ml = ml  # use maximum likelihood method
        if self.ml:
            assert self.num_gen == 1 and self.num_disc == 1 and self.num_mcmc == 1, "invalid settings for ML training"

        self.noise_std = np.sqrt(2 * self.alpha * self.eta)
        self.build_bgan_graph()
        print 'Model initialization has gone successfully!'

    def initialize_wgts(self, scope_str):
        # define p(theta_x)
        param_list = []
        if scope_str == "generator":
            for gi in range(self.num_gen):
                for m in range(self.num_mcmc):
                    # get ordered dict representing params of current generator
                    # initialization mode (params == None)
                    wgts_ = self.generator(tf.constant(0.0, dtype=tf.float32, shape=self.lr_sampler.shape),
                                           params=None, return_params=True, mask='_%04d_%04d' % (gi, m))[-1]

                    #new_keys = map(lambda x: x + '_%04d_%04d' % (gi, m), wgts_.keys())
                    #wgts_ = dict(zip(new_keys, wgts_.values()))
                    param_list.append(wgts_)

        elif scope_str == "discriminator":
            for di in range(self.num_disc):
                for m in range(self.num_mcmc):
                    wgts_ = self.discriminator(tf.constant(0.0, dtype=tf.float32, shape=self.hr.shape),
                                               params=None, return_params=True, mask='_%04d_%04d' % (di, m))[-1]

                    #new_keys = map(lambda x: x + '_%04d_%04d' % (di, m), wgts_.keys())
                    #wgts_ = dict(zip(new_keys, wgts_.values()))
                    param_list.append(wgts_)
        else:
            raise RuntimeError("invalid scope!")

        return param_list

    def build_bgan_graph(self):
        #self.hr = tf.placeholder(tf.float32, [self.batch_size] + self.hr_dim, name='real_images')
        # define placeholder for LR images
        #self.lr = tf.placeholder(tf.float32, [self.batch_size] + self.lr_dim + [self.num_gen], name='lr_images')
        self.lr_sampler = tf.placeholder(tf.float32, [self.batch_size] + self.lr_dim, name='lr_sampler')

        # initialize generator weights
        self.gen_param_list = self.initialize_wgts("generator")  # p(theta_g)
        self.disc_param_list = self.initialize_wgts("discriminator")  # p(theta_d)
        # build discrimitive losses and optimizers
        # prep optimizer args
        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])  # adaptive learning rate

        # compile all disciminative weights
        t_vars = tf.trainable_variables()
        self.d_vars = []
        for di in xrange(self.num_disc):
            for m in xrange(self.num_mcmc):
               self.d_vars.append([var for var in t_vars if 'd_' in var.name and "_%04d_%04d" % (di, m) in var.name])

        # build disc losses and optimizers
        self.d_losses, self.d_optims, self.d_optims_adam = [], [], []

        for di, disc_params in enumerate(self.disc_param_list):
            # for each setting of theta_d we count losses using each setting of theta_g
            # build loss for each theta_d
            d_logits, _ = self.discriminator(self.hr, disc_params)
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                                                 labels=tf.ones_like(d_logits)))
            d_loss_fakes = []
            for gi, gen_params in enumerate(self.gen_param_list):
                sample = self.lr[:, :, :, :, gi % self.num_gen]
                d_logits_, _ = self.discriminator(self.generator(sample, gen_params), disc_params)

                d_loss_fake_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                                                      labels=tf.zeros_like(d_logits_)))
                d_loss_fakes.append(d_loss_fake_)

            d_losses = []
            for d_loss_fake_ in d_loss_fakes:
                d_loss_ = d_loss_real * float(self.num_gen) + d_loss_fake_
                if not self.ml:
                    d_loss_ += self.disc_prior(disc_params) + self.disc_noise(disc_params)
                d_losses.append(tf.reshape(d_loss_, [1]))

            d_loss = tf.reduce_logsumexp(tf.concat(d_losses, 0))
            self.d_losses.append(d_loss)
            d_opt = self._get_optimizer(self.d_learning_rate)
            self.d_optims.append(d_opt.minimize(d_loss, var_list=self.d_vars[di]))
            d_opt_adam = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=0.5)
            self.d_optims_adam.append(d_opt_adam.minimize(d_loss, var_list=self.d_vars[di]))

        # build generative losses and optimizers
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.g_vars = []
        for gi in xrange(self.num_gen):
            for m in xrange(self.num_mcmc):
                self.g_vars.append([var for var in t_vars if 'g_' in var.name and "_%04d_%04d" % (gi, m) in var.name])

        self.g_losses, self.g_optims, self.g_optims_adam = [], [], []
        for gi, gen_params in enumerate(self.gen_param_list):

            gi_losses = []
            for disc_params in self.disc_param_list:
                gen_sample = self.generator(self.lr[:, :, :, :, gi % self.num_gen], gen_params)
                d_logits_, d_features_fake = self.discriminator(gen_sample, disc_params)

                # calculate features using perceptual mode vgg22/vgg54
                if FLAGS.perceptual_mode == 'VGG22':
                    with tf.name_scope('vgg19_1') as scope:
                        extracted_feature_gen = VGG19_slim(gen_sample, 'VGG22', reuse=tf.AUTO_REUSE, scope=scope)
                    with tf.name_scope('vgg19_2') as scope:
                        extracted_feature_target = VGG19_slim(self.hr, 'VGG22', reuse=tf.AUTO_REUSE, scope=scope)

                elif FLAGS.perceptual_mode == 'VGG54':
                    with tf.name_scope('vgg19_1') as scope:
                        extracted_feature_gen = VGG19_slim(gen_sample, 'VGG54', reuse=tf.AUTO_REUSE, scope=scope)
                    with tf.name_scope('vgg19_2') as scope:
                        extracted_feature_target = VGG19_slim(self.hr, 'VGG54', reuse=tf.AUTO_REUSE, scope=scope)

                else:
                    extracted_feature_gen = gen_sample
                    extracted_feature_target = self.hr

                diff = extracted_feature_gen - extracted_feature_target
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
                _, d_features_real = self.discriminator(self.hr, disc_params)

                g_loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                                                 labels=tf.ones_like(d_logits_)))
                g_loss_ += tf.reduce_mean(huber_loss(d_features_real[-1], d_features_fake[-1]))
                g_loss_ += content_loss
                if not self.ml:
                    g_loss_ += self.gen_prior(gen_params) + self.gen_noise(gen_params)
                gi_losses.append(tf.reshape(g_loss_, [1]))

            g_loss = tf.reduce_logsumexp(tf.concat(gi_losses, axis=0))
            self.g_losses.append(g_loss)
            g_opt = self._get_optimizer(self.g_learning_rate)
            self.g_optims.append(g_opt.minimize(g_loss, var_list=self.g_vars[gi]))
            g_opt_adam = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
            self.g_optims_adam.append(g_opt_adam.minimize(g_loss, var_list=self.g_vars[gi]))

        # build samplers
        self.gen_samplers = []
        for gi, gen_params in enumerate(self.gen_param_list):
            self.gen_samplers.append(self.generator(self.lr_sampler, gen_params))

    def _get_optimizer(self, lrate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lrate, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lrate, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

    @staticmethod
    def generator(features, params=None, return_params=False, mask='_0000_0000'):
        # Upside-down all-convolutional resnet

        mapsize = 3

        # See Arxiv 1603.05027
        scope = 'g'
        model = Model(scope, features, params, mask)

        for j in range(2):
            model.add_residual_block(256, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        # don't apply second upscaling (we need 2x factor)
        #model.add_upscale()
        model.add_pixel_shuffler(2)  # instead of upscaling
        model.add_conv2d(256, mapsize=mapsize, stride=1, stddev_factor=1.)
        model.add_batch_norm()
        model.add_relu()

        model.add_conv2d_transpose(256, mapsize=mapsize, stride=1, stddev_factor=1.)

        # Finalization a la "all convolutional net"

        model.add_conv2d(96, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()

        model.add_relu()

        model.add_conv2d(96, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        # Last layer is sigmoid with no batch normalization
        model.add_conv2d(3, mapsize=1, stride=1, stddev_factor=1.)
        model.add_sigmoid()

        gene_vars = model.params
        #   model.summary()
        if return_params:
            return model.get_output(), gene_vars
        return model.get_output()

    @staticmethod
    def discriminator(disc_input, params=None, return_params=False, mask='_0000_0000'):
        mapsize = 3
        layers = [64, 128, 256, 512]

        model = Model('d', 2 * disc_input - 1, params, mask)  # scale between -1 and 1

        for layer in range(len(layers)):
            nunits = layers[layer]
            stddev_factor = 2.0

            model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
            model.add_batch_norm()
            model.add_relu()

        # Finalization a la "all convolutional net"
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        # Linearly map to real/fake and return average score
        # (softmax will be applied later)
        model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
        model.add_flatten()
        model.add_dense(100, stddev_factor=2)
        model.add_relu()
        h_end = model.get_output()
        model.add_dense(1)
        h_out = model.get_output()

        disc_vars = model.params
        # model.summary()
        if return_params:
            return h_out, h_end, disc_vars
        return h_out, h_end

    def gen_prior(self, gen_params):
        prior_loss = 0.0
        for var in gen_params.values():
            nn = tf.divide(var, self.prior_std)
            prior_loss += tf.reduce_mean(tf.multiply(nn, nn))

        prior_loss /= self.dataset_size
        return prior_loss

    def gen_noise(self, gen_params):
        noise_loss = 0.0
        for name, var in gen_params.iteritems():
            noise_ = tf.distributions.Normal(loc=0., scale=self.noise_std * tf.ones(var.get_shape()))
            noise_loss += tf.reduce_sum(var * noise_.sample())

        noise_loss /= self.dataset_size
        return noise_loss

    def disc_prior(self, disc_params):
        prior_loss = 0.0
        for var in disc_params.values():
            nn = tf.divide(var, self.prior_std)
            prior_loss += tf.reduce_mean(tf.multiply(nn, nn))

        prior_loss /= self.dataset_size
        return prior_loss

    def disc_noise(self, disc_params):
        noise_loss = 0.0
        for var in disc_params.values():
            noise_ = tf.distributions.Normal(loc=0., scale=self.noise_std * tf.ones(var.get_shape()))
            noise_loss += tf.reduce_sum(var * noise_.sample())

        noise_loss /= self.dataset_size
        return noise_loss
