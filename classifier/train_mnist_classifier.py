import tensorflow as tf
from time import localtime, strftime
import os

from tensorflow.examples.tutorials.mnist import input_data
import utils.cnn_utils as cnn
import classifier.const_cls as const


tf.logging.set_verbosity(tf.logging.ERROR)  # Show Tensorflow error only
mnist = input_data.read_data_sets(os.path.join(".", "..", "MNIST_data"), one_hot=True)


def mnist_classifier(input, keep_prob):
    """
    Define a MNIST classifier that contains 2 convolutional layers and 2 fully connected layers
    with dropout.

    :param input: input tensor
    :return: softmax probability of built CNN
    """

    # Classifier architecture
    with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):

        # Convolutional layer 1
        W_conv1 = cnn.weight_variable([5, 5, 1, 32], name="W_conv1")
        b_conv1 = cnn.bias_variable([32], name="b_conv1")

        h_conv1 = tf.nn.relu(cnn.conv2d_basic(input, W_conv1, b_conv1), name="h_conv1")
        h_pool1 = cnn.max_pool_2x2(h_conv1)

        # Convolutional layer 2
        W_conv2 = cnn.weight_variable([5, 5, 32, 64], name="W_conv2")
        b_conv2 = cnn.bias_variable([64], name="b_conv2")

        h_conv2 = tf.nn.relu(cnn.conv2d_basic(h_pool1, W_conv2, b_conv2), name="h_conv2")
        h_pool2 = cnn.max_pool_2x2(h_conv2)

        # Fully connected layer 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        W_fc1 = cnn.weight_variable([7 * 7 * 64, 1024], name="W_fc1")
        b_fc1 = cnn.bias_variable([1024], name="b_fc1")

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

        # Fully connected layer 2 (Output layer)
        W_fc2 = cnn.weight_variable([1024, 10], name="W_fc2")
        b_fc2 = cnn.bias_variable([10], name="b_fc2")

        y_ = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_')

        return y_


def main(argv=None):

    with tf.device('/gpu:0'):

        # Input layer
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        print("Finished building inputs and outputs")

        # Build nn model
        y_ = mnist_classifier(x_image, keep_prob)
        print("Finished building MNIST classifier")

        # Evaluation metrics
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]), name="cross_entropy")
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        print("Finished building metrics")

        # Training algorithm
        train_step = tf.train.AdamOptimizer(const.L_RATE).minimize(cross_entropy)
        print("Finished building training operation")

        # Log operation
        log_accuracy = tf.summary.scalar("accuracy", accuracy)

        # GPU memory tricks, so that computation doesn't take all GPU resource at once
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Set up model saving
            model_dir = os.path.join(".", "model")

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_saver = tf.train.Saver(max_to_keep=10)

            # Set up logger
            log_dir = os.path.join(".", "log", strftime("%Y-%m-%d-%H%M%S", localtime()))

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            summary_writer = tf.summary.FileWriter(log_dir)

            # == Train classifier == #

            keep_probab = 0.5

            curr_epoch = 0
            max_step = int((mnist.train.images.shape[0] * const.MAX_EPOCH) / const.BATCH_SIZE)
            for step in range(max_step):

                if (step * const.BATCH_SIZE) % mnist.train.images.shape[0] == 0:

                    # Log and report accuracy
                    entropy, accu, summary_str = sess.run([cross_entropy, accuracy, log_accuracy],
                                                 feed_dict={x: mnist.validation.images,
                                                            y: mnist.validation.labels, keep_prob: 1.0})
                    summary_writer.add_summary(summary_str, step)
                    print("Step = {}, epoch = {}, validation accuracy = {}, entropy = {}".format(step, curr_epoch, accu, entropy))

                    # Save model weights
                    model_saver.save(sess, os.path.join(model_dir, "mnist_classifier.ckpt"), step)
                    curr_epoch += 1

                batch_xs, batch_ys = mnist.train.next_batch(const.BATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: const.KEEP_PROB})

            entropy, accu, summary_str = sess.run([cross_entropy, accuracy, log_accuracy],
                                         feed_dict={x: mnist.validation.images,
                                                    y: mnist.validation.labels, keep_prob: 1.0})
            summary_writer.add_summary(summary_str, max_step)
            print("Step = {}, epoch = {}, validation accuracy = {}, entropy = {}".format(max_step, const.MAX_EPOCH, accu, entropy))
            model_saver.save(sess, os.path.join(model_dir, "mnist_classifier.ckpt"), max_step)
            print("Finished training")

            print("On test set:")
            entropy, accu = sess.run([cross_entropy, accuracy], feed_dict={x: mnist.test.images,
                                                                           y: mnist.test.labels, keep_prob: 1.0})
            print(
                "Epoch = {}, test accuracy = {}, entropy = {}".format(const.MAX_EPOCH, accu, entropy))


if __name__ == "__main__":
    tf.app.run()
