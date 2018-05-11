import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from glob import glob

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load pretrained vgg16 model and weights 
    # note that fc6 and fc7 are converted from fc to conv layers
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get (image_input, keep_prob, layer3_out, layer4_out, layer7_out) tensors
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # decoder 1
    with tf.name_scope("decoder1"):
        # upsample layer7 by 2 
        # if input is 224x224: (7x7xx4096) => (14x14xnum_classes)
        input = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes,4,
         strides=(2,2), padding='SAME')

        # add skip connection
        # first convert pool4 to output dimensions of num_classes by adding a 1x1 conv
        pool4_11 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
         strides=(1,1), padding='SAME')
        decoder1 = tf.add(pool4_11, input) 
    
    # decoder 2
    with tf.name_scope("decoder2"):
        # upsample by 2
        # (14x14xnum_classes) => (28x28xnum_classes)
        input = tf.layers.conv2d_transpose(decoder1, num_classes, 4,
         strides=(2,2), padding='SAME')

        # add skip connection
        # first convert pool3 to output dimensions of num_classes by adding a 1x1 conv
        pool3_11 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
         strides=(1,1), padding='SAME')
        decoder2 = tf.add(pool3_11, input)

    # output
    with tf.name_scope("output"):
        output = tf.layers.conv2d_transpose(decoder2, num_classes, 16,
         strides=(8,8), padding='SAME')

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    with tf.name_scope("xent"):
        logits = cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct_label,
        logits=nn_last_layer)

    with tf.name_scope("loss"):
        cross_entropy_loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    # initialize global variables
    sess.run(tf.global_variables_initializer())

    print('training...')

    for e in range(epochs):
        print("epoch: {}".format(e))

        for images, ground_truth in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
             feed_dict={input_image: images,
             correct_label: ground_truth,
             keep_prob: 0.5, learning_rate: 0.001})

            print("epoch {}: training loss {:.4f}".format(e, loss))

    print('training done!')

    # pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    input_image = tf.placeholder(tf.float32, shape=[None, *image_shape, 3])
    correct_label = tf.placeholder(tf.float32, shape=[None, *image_shape, num_classes])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    epochs = 1
    batch_size = 289

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir,
         'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        #  Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        _, train_op, cross_entropy_loss = optimize(output, correct_label,
         learning_rate, num_classes)
        
        #  Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
         input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
         helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, output, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
