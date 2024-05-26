# coding=utf-8
"""Implementation of ALDA attack."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import scipy.stats as st
from imageio import imread, imsave
from func import *
import pandas as pd
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import random
import warnings


warnings.filterwarnings('ignore', category=UserWarning)

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 8, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('portion', 0.5, 'protion for the mixed image, current best: 0.5')

tf.flags.DEFINE_float('gus_mean', 0.3, 'The mean value of Gaussian noise during data augmentation.')

tf.flags.DEFINE_float('gus_std', 0.5, 'The variance value of Gaussian noise during data augmentation.')

tf.flags.DEFINE_integer('sigma', 9, "The number for the  ")

tf.flags.DEFINE_integer('patch_size', 50, 'image patch size')

tf.flags.DEFINE_integer('mask_percent', 55, 'cam mask percent for look ahead')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models', 'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs', 'Output directory with images.')

tf.flags.DEFINE_string('model_name', "inception_v3", "Name of the model")

tf.flags.DEFINE_string('attack_method', "", "Name of the model")




FLAGS = tf.flags.FLAGS
print(f'model_name = {FLAGS.model_name}')
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)


model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


_layer_names = {"resnet_v2": ["PrePool", "Predictions"],
                "inception_v3": ["PrePool", "Predictions"],
                "inception_v4": ["PrePool", "Predictions"],
                "inception_resnet_v2": ["PrePool", "Predictions"],
               }


model_variables_map = {"resnet_v2": ["resnet_v2", "resnet_v2"],
                       "inception_v3": ["InceptionV3", "inception_v3"],
                       "inception_v4": ["InceptionV4", "inception_v4"],
                       "inception_resnet_v2": ["InceptionResnetV2", "inception_resnet_v2"],
                       }


percentile_list = list(range(1, FLAGS.sigma + 1))


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


if 'ti' in FLAGS.attack_method.lower():
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0

        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # imsave(f, img_as_ubyte((images[i, :, :, :] + 1.0) * 0.5), format='png')
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def grad_cam(end_points, predicted_class, nb_classes=1001, eval_image_size=FLAGS.image_height):
    _logits_name = "Logits"
    # Conv layer tensor [?,10,10,2048]
    layer_name = _layer_names[FLAGS.model_name][0]
    conv_layer = end_points[layer_name]
    predicted_class = tf.reshape(predicted_class, [-1])
    one_hot = tf.one_hot(predicted_class, nb_classes, 1.0)

    signal = tf.multiply(end_points[_logits_name], one_hot)
    loss = tf.reduce_mean(signal, axis=1)
    grads = tf.gradients(loss, conv_layer)[0]

    # Normalizing the gradients
    norm_grads = tf.divide(grads, tf.reshape(tf.sqrt(tf.reduce_mean(tf.square(grads), axis=[1, 2, 3])), [FLAGS.batch_size, 1, 1, 1]) + tf.constant(1e-5))

    output, grads_val = conv_layer, norm_grads # sess.run([conv_layer, norm_grads], feed_dict={x_input: images})
    weights = tf.reduce_mean(grads_val, axis=(1, 2)) 			 # [8, 2048]
    cam = tf.ones(output.shape[0: 3], dtype=tf.float32)	 # [10,10]

    cam += tf.einsum("bmnk, bkl->bmn", output, tf.reshape(weights, [weights.shape[0], weights.shape[1], 1]))
    cam = tf.maximum(cam, 0)
    cam = tf.divide(cam, tf.reshape(tf.reduce_max(cam, axis=[1, 2]), [FLAGS.batch_size, 1, 1]))
    cam = tf.image.resize_images(tf.expand_dims(cam, axis=-1), [eval_image_size, eval_image_size], method=0)
    return cam


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def swap_elements(tensor, idx1_tensor, idx2_tensor):
    # 使用tf.gather操作获取需要交换的元素
    element1 = tf.gather(tensor, idx1_tensor)
    # element2 = tf.gather(tensor, idx2_tensor)
    element2 = tf.zeros_like(element1)
    # 构建一个新的张量，交换元素的位置
    return tf.tensor_scatter_nd_update(tensor, [[idx1_tensor], [idx2_tensor]], [element2, element1])


def swap_max_min(cam_value, change_value, percent=1):
    values = tf.concat(cam_value, axis=0)
    # 对值进行排序
    sorted_values, indices = tf.math.top_k(values, k=tf.shape(values)[0])
    total_len = indices.shape[0] * percent

    for index in range(total_len // 20):
        begin_pos = indices[index]
        end_pos = indices[-(1 + index)]
        change_value = swap_elements(change_value, begin_pos, end_pos)

    return change_value


def shuffler_single_pic(cam, input_data, patch_size, eval_image_size, percent=1):
    begin_x, begin_y = pair(int((eval_image_size % patch_size) / 2))
    number_patch = eval_image_size // patch_size
    # patch_info = dict()
    cam_list, patch_list = list(), list()
    for out_loop in range(number_patch):
        for inner_loop in range(number_patch):
            start_x = begin_x + inner_loop * patch_size
            start_y = begin_y + out_loop * patch_size
            patch_cam = tf.image.crop_to_bounding_box(cam, start_y, start_x, patch_size, patch_size)
            patch_data = tf.image.crop_to_bounding_box(input_data, start_y, start_x, patch_size, patch_size)
            cam_value = tf.expand_dims(tf.reduce_sum(patch_cam), axis=0)
            cam_list.append(cam_value)
            patch_list.append(tf.expand_dims(patch_data, axis=0))

    # 影响力排名首尾替换
    shuffler_patch = swap_max_min(cam_list, tf.concat(patch_list, axis=0), percent=percent)

    patch_data_list = tf.split(shuffler_patch, shuffler_patch.shape[0], axis=0)
    patch_data_list = [tf.squeeze(data, axis=0) for data in patch_data_list]
    shuffler_data = None
    ori_data = tf.image.crop_to_bounding_box(input_data, begin_y, begin_x,
                                             number_patch * patch_size, number_patch * patch_size)
    ori_data = tf.image.pad_to_bounding_box(ori_data, begin_y, begin_x, eval_image_size, eval_image_size)
    ori_data = input_data - ori_data
    for out_loop in range(number_patch):
        for inner_loop in range(number_patch):
            start_x = begin_x + inner_loop * patch_size
            start_y = begin_y + out_loop * patch_size
            index = out_loop * number_patch + inner_loop

            batch_data = tf.image.pad_to_bounding_box(patch_data_list[index], start_y, start_x, eval_image_size,
                                                      eval_image_size)
            if shuffler_data is None:
                shuffler_data = ori_data + batch_data
                continue
            shuffler_data += batch_data
    return shuffler_data


def shuffler_single_cam(cam, input_data, patch_size, eval_image_size, percent=1):
    begin_x, begin_y = pair(int((eval_image_size % patch_size) / 2))
    number_patch = eval_image_size // patch_size
    cam_list, patch_list = list(), list()

    for out_loop in range(number_patch):
        for inner_loop in range(number_patch):
            start_x = begin_x + inner_loop * patch_size
            start_y = begin_y + out_loop * patch_size

            patch_cam = tf.image.crop_to_bounding_box(cam, start_y, start_x, patch_size, patch_size)
            cam_value = tf.expand_dims(tf.reduce_sum(patch_cam), axis=0)
            cam_list.append(cam_value)
            patch_list.append(tf.expand_dims(patch_cam, axis=0))

    # 影响力排名首尾替换
    shuffer_patch = swap_max_min(cam_list, tf.concat(patch_list, axis=0), percent=percent)

    patch_data_list = tf.split(shuffer_patch, shuffer_patch.shape[0], axis=0)
    patch_data_list = [tf.squeeze(data, axis=0) for data in patch_data_list]
    shuffer_data = None
    ori_data = tf.image.crop_to_bounding_box(cam, begin_y, begin_x,
                                             number_patch * patch_size, number_patch * patch_size)
    ori_data = tf.image.pad_to_bounding_box(ori_data, begin_y, begin_x, eval_image_size, eval_image_size)
    ori_data = cam - ori_data
    for out_loop in range(number_patch):
        for inner_loop in range(number_patch):
            start_x = begin_x + inner_loop * patch_size
            start_y = begin_y + out_loop * patch_size
            index = out_loop * number_patch + inner_loop

            batch_data = tf.image.pad_to_bounding_box(patch_data_list[index], start_y, start_x, eval_image_size,
                                                      eval_image_size)
            if shuffer_data is None:
                shuffer_data = ori_data + batch_data
                continue
            shuffer_data += batch_data
    return shuffer_data


def attention_shuffler(input_data, cam, patch_size=FLAGS.patch_size, eval_image_size=FLAGS.image_height, percent=1):
    shuffler_list = list()
    batch_size = input_data.shape[0]
    for index in range(batch_size):
        shuffler_pic = shuffler_single_cam(cam[index], input_data[index], patch_size, eval_image_size, percent=percent)
        shuffler_list.append(tf.expand_dims(shuffler_pic, axis=0))
    return tf.concat(shuffler_list, axis=0)


def get_model_results(x, model_name=FLAGS.model_name, num_classes=1001):
    if model_name == 'resnet_v2':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_v3':
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_v4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_resnet_v2':
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    else:
        logits, end_points = None, None

    return logits, end_points


def percentile_value(x, down_p):
    '''
    :param x:  [batch_size, 299, 299, 1]
    :param down_p: 70
    :return: [batch_size, 299, 299, 1]
    '''
    down_percentile = tf.contrib.distributions.percentile(x, down_p, axis=[1, 2, 3])
    down_v = tf.minimum(tf.sign(x - tf.reshape(down_percentile, [FLAGS.batch_size, 1, 1, 1])), 0)
    # down_v = tf.maximum(tf.sign(x - tf.reshape(down_percentile, [FLAGS.batch_size, 1, 1, 1])), 0)
    return x * down_v


def attention_aug(x, cam_value):
    eps = 2.0 * FLAGS.max_epsilon / 255.0

    augment_value = [tf.random.uniform(x.get_shape().as_list(), minval=-eps, maxval=eps, dtype=tf.dtypes.float32)
                     + (1 - FLAGS.portion) * x
                     + FLAGS.portion * (tf.random.normal(x.get_shape().as_list()[1:],
                                                         mean=FLAGS.gus_mean,
                                                         stddev=FLAGS.gus_std,
                                                         dtype=tf.dtypes.float32)
                                        * attention_shuffler(x, cam_value, patch_size=FLAGS.patch_size, percent=p))
                     for p in percentile_list]

    return tf.concat(augment_value, axis=0)


def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    logits_v3, end_points_inc_v3 = get_model_results(x=x, model_name=FLAGS.model_name, num_classes=num_classes)
    cam_value = grad_cam(end_points=end_points_inc_v3, predicted_class=y)

    cam_p = percentile_value(cam_value, down_p=FLAGS.mask_percent)
    print(f'cam_p={cam_p.shape}')
    x_nes = x + alpha * cam_p * grad

    aug_data = attention_aug(x_nes, cam_value)

    x_batch = tf.concat([aug_data, aug_data / 2., aug_data / 4., aug_data / 8., aug_data / 16.], axis=0)
    num_size = len(percentile_list)

    '''input diversity'''
    x_input = x_batch
    if 'di' in FLAGS.attack_method.lower():
        x_input = input_diversity(x_batch)

    logits_v3, end_points_inc_v3 = get_model_results(x=x_input, model_name=FLAGS.model_name, num_classes=num_classes)
    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5 * num_size, axis=0)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    noise = tf.reduce_mean(tf.split(tf.gradients(cross_entropy, x_batch, colocate_gradients_with_ops=True)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)
    noise = tf.reduce_sum(tf.split(noise, num_size), axis=0)

    '''TI'''
    if 'ti' in FLAGS.attack_method.lower():
        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return i < num_iter


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.ERROR)

    save_path = os.path.join(FLAGS.output_dir,
                             'single_attack',
                             f'model_{FLAGS.model_name}_method_{FLAGS.attack_method}_gus_{FLAGS.gus_mean}_{FLAGS.gus_std}_patch_{FLAGS.patch_size}_sigma_{FLAGS.sigma}_portion_{FLAGS.portion}_mask_{FLAGS.mask_percent}')
    check_or_create_dir(save_path)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        logits_v3, end_points_v3 = get_model_results(x=x_input, model_name=FLAGS.model_name, num_classes=1001)
        pred = tf.argmax(end_points_v3['Predictions'], 1)

        i = tf.constant(0, dtype=tf.float32)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, loop_vars=[x_input, pred, i, x_max, x_min, grad])

        # Run computation
        s = tf.train.Saver(slim.get_model_variables(scope=model_variables_map[FLAGS.model_name][0]))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s.restore(sess, model_checkpoint_map[model_variables_map[FLAGS.model_name][1]])
            idx = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, save_path)


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
