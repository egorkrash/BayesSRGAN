import random
import numpy as np
import os
import tensorflow as tf
from arch import BSRGAN
import time
import json
from utils import print_images, load_weights, setup_vgg
from scipy.misc import imsave
# TODO: support of multiple generators
# TODO: fix saving samples (random choice replace!=True)
# TODO: !!! fix calling sess.run(features/labels), they MUST be called at the same time !!! IMPORTANT!!!
# Configuration (alphabetically)

tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 5,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'demo',
                           "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', .90,
                          "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_decay', 3.0, 'lr decay')

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('num_gen', 1, 'number of generators')

tf.app.flags.DEFINE_integer('num_disc', 1, 'number of discriminators')

tf.app.flags.DEFINE_integer('num_mcmc', 1, 'number of mcmc')

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_bool('save_weights', True, 'save weights or not')

tf.app.flags.DEFINE_bool('save_samples', True, 'save samples or not')

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")

tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_iter', 1000, 'number of training iterations')

tf.app.flags.DEFINE_integer('train_time', 20,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_float('vgg_scaling', 0.0061, 'weight of accepting vgg features')

tf.app.flags.DEFINE_string('perceptual_mode', 'VGG54', 'perceptual mode to extract features for additive loss')

tf.app.flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')

tf.app.flags.DEFINE_bool('continue_training', True, 'continue training process or not')
FLAGS = tf.app.flags.FLAGS


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer


def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):
    if image_size is None:
        image_size = FLAGS.sample_size

    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([64, 64, channels])

    # Crop and other random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, .95, 1.05)
    image = tf.image.random_brightness(image, .05)
    image = tf.image.random_contrast(image, .95, 1.05)
    image = tf.reshape(image, [1, 64, 64, 3])
    image = tf.cast(image, tf.float32) / 255.0

    K = 2  # upscaling factor
    downsampled = tf.image.resize_area(image, [image_size / K, image_size / K])

    feature = tf.reshape(downsampled, [image_size / K, image_size / K, 3])
    label = tf.reshape(image, [image_size, image_size, 3])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity=capacity_factor * FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)

    return features, labels


def main():
    tf.reset_default_graph()
    sess, summary_writer = setup_tensorflow()
    train(sess)


def train(sess):
    filenames = map(lambda x: os.path.join(FLAGS.dataset, x), os.listdir(FLAGS.dataset))
    features, labels = setup_inputs(sess, filenames)

    dataset_size = len(filenames)
    batch_size = FLAGS.batch_size
    start_iter = 0  # in case we don't load any weights

    num_gen = FLAGS.num_gen
    num_disc = FLAGS.num_disc
    num_mcmc = FLAGS.num_mcmc

    # ==================================================================================================================
    bsrgan = BSRGAN(hr_images=labels, lr_images=tf.reshape(features, [batch_size, 32, 32, 3, 1]),
                    dataset_size=dataset_size, J=num_gen,
                    J_d=num_disc, M=num_mcmc, batch_size=batch_size)

    setup_vgg(sess)
    sess.run(tf.global_variables_initializer())

    if FLAGS.continue_training:
        start_iter = load_weights(sess)
    # ==================================================================================================================

    num_train_iter = FLAGS.train_iter

    optimizer_dict = {"disc": bsrgan.d_optims_adam, "gen":  bsrgan.g_optims_adam}

    base_learning_rate = FLAGS.learning_rate_start  # for now we use same learning rate for Ds and Gs
    lr_decay_rate = FLAGS.learning_rate_decay
    print 'Starting training process...'
    for train_iter in range(start_iter, start_iter+num_train_iter):

        if train_iter == 5000:
            print("Switching to user-specified optimizer")
            optimizer_dict = {"disc": bsrgan.d_optims_adam,
                              "gen": bsrgan.g_optims_adam}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter * batch_size) / float(dataset_size)))

        # compute disc losses
        disc_info = sess.run(optimizer_dict["disc"] + bsrgan.d_losses, feed_dict={bsrgan.d_learning_rate: learning_rate})

        d_losses = disc_info[num_disc:num_disc * 2]

        # compute generative losses
        gen_info = sess.run(optimizer_dict["gen"] + bsrgan.g_losses, feed_dict={bsrgan.g_learning_rate: learning_rate})
        g_losses = [g_ for g_ in gen_info if g_ is not None]

        if train_iter > 0 and train_iter % FLAGS.checkpoint_period == 0:

            print("Iter %i" % train_iter)
            print("Disc losses = %s" % (", ".join(["%.2f" % dl for dl in d_losses])))
            print("Gen losses = %s" % (", ".join(["%.2f" % gl for gl in g_losses])))

            print("saving results and samples")

            results = {"disc_losses": map(float, d_losses),
                       "gen_losses": map(float, g_losses),
                       "timestamp": time.time()}

            with open(os.path.join(FLAGS.checkpoint_dir, 'results_%i.json' % train_iter), 'w') as fp:
                json.dump(results, fp)

            if FLAGS.save_samples:
                for zi in xrange(num_gen):
                    _imgs, _ps = [], []
                    for _ in range(10):
                        lr, hr = sess.run([features, labels])  # run both to keep correspondence
                        sampled_imgs = sess.run(bsrgan.gen_samplers[zi * num_mcmc],
                                                feed_dict={bsrgan.lr_sampler: lr})

                        _imgs.append(sampled_imgs)
                    sampled_imgs = np.concatenate(_imgs)

                    print_images(sampled_imgs, "BSRGAN_%i_%.2f" % (zi, g_losses[zi * num_mcmc]),
                                 train_iter, directory=FLAGS.checkpoint_dir)

                imsave(FLAGS.checkpoint_dir + '/lr_%i.png' % train_iter, lr)  # check correspondence
                imsave(FLAGS.checkpoint_dir + '/hr_%i.png' % train_iter, hr)
                # print_images(sess.run(labels), "RAW", train_iter, directory=FLAGS.checkpoint_dir)

                if FLAGS.save_weights:
                    var_dict = {}
                    for var in tf.trainable_variables():
                        var_dict[var.name] = sess.run(var.name)

                    np.savez_compressed(os.path.join(FLAGS.checkpoint_dir, "weights_%i.npz" % train_iter), **var_dict)

                print("done")

if __name__ == '__main__':
    main()
