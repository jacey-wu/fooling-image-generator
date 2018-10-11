import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import localtime, strftime
import os

from tensorflow.examples.tutorials.mnist import input_data
from classifier.train_mnist_classifier import mnist_classifier
import utils.cnn_utils as cnn
import adversary.const_adv as const


tf.logging.set_verbosity(tf.logging.ERROR)  # Show Tensorflow error only

# Import MNIST data set
mnist = input_data.read_data_sets(os.path.join(".", "..", "MNIST_data"), one_hot=True)


def get_vl_samples_with_class_c(c=2, max_n=10):
    """
    Obtain max_n number of samples with class label c from the validation set of MNIST data set.

    If validation set contains less than max_n samples with class label c, all samples with class
    label c will be returned only. Train/validation/test splits are the same as when MNIST
    classifier is trained.

    :param c: desired class label
    :param max_n: number of samples drawn from the validation set
    :return: 2D numpy array (n, h * w) containing max_n input images
    """

    ims = mnist.validation.images
    lbs = mnist.validation.labels

    idx = [sample[c] == 1 for sample in lbs]

    ims_c = ims[idx][:]
    return ims_c[0:min(max_n, ims_c.shape[0])][:]


def format_output(ims, deltas, ims_alt):
    """
    Combine and format original images, delta maps, and altered images into a 2D list where the
    rows are the number of samples, and the columns are original images, deltas, and altered
    adversarial images.

    :param ims: 4D numpy array containing input images with format (n, h, w, d)
    :param deltas: 4D numpy array containing delta maps changing input images to altered images with
                format (n, h, w, d)
    :param ims_alt: 4D numpy array containing altered images with format (n, h, w, d)
    :return: A 2D list
    """

    ims = np.squeeze(ims, axis=3)
    deltas = np.squeeze(deltas, axis=3)
    ims_alt = np.squeeze(ims_alt, axis=3)
    rst = []
    for i in range(ims.shape[0]):
        rst.append([ims[i], deltas[i], ims_alt[i]])
    return rst


def visualize_result(rst):
    """
    Visualize the 2D list result rst. Different rows display different samples, the columns display the
    original image, delta map, and adversarial image of one sample.

    :param rst: 2D list result to be visualized
    :return: None
    """

    rows = len(rst)

    # Simple error checking
    if rows < 1:
        print("Error: result to visualize is empty")
        return

    cols = len(rst[0])
    fig = plt.figure()
    for r in range(rows):
        for c in range(cols):
            sub_im = rst[r][c]
            fig.add_subplot(rows, cols, r * cols + c + 1)
            plt.axis("off")
            plt.imshow(sub_im, cmap="Greys")
    plt.show()


def save_adversarial_images(rst, name_suffixes=["orig", "delta", "adv"], format="png"):
    """
    Save original images, delta maps, and adversarial images to "./saved_images/timestamp_dir".

    :param rst: the 2D list result that contains original images, delta maps, and adversarial images
    :param name_suffixes: suffixes added to the file names of the different types of outputs
    :param format: the format that outputs are saved to
    :return: None
    """

    # Set up dic for saving
    save_dir = os.path.join('.', 'saved_images', strftime("%Y-%m-%d-%H%M%S", localtime()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Simple error checking
    rows = len(rst)
    if rows < 1:
        print("Error: result to visualize is empty")
        return
    if len(name_suffixes) < len(rst[0]):
        print("Error: not enough file name suffixes provided for different output image types")

    # Save images
    for i in range(rows):
        for j in range(len(rst[0])):
            file_name = "{}_{}.{}".format(i, name_suffixes[j], format)
            plt.imsave(os.path.join(save_dir, file_name), rst[i][j], cmap="Greys")
    print("Saved adversarial images at " + save_dir)


def main(argv=None):
    """
    The script to pick images of digit '2' from MNIST data set, which are correctly classified as '2' by
    the trained model MNIST classifier, and modify them so the network incorrectly classifies them as '6'
    """

    with tf.device('/gpu:0'):

        # Input layer
        x = tf.placeholder(tf.float32, shape=[const.BATCH_SIZE, 784], name="x")
        y = tf.placeholder(tf.float32, shape=[const.BATCH_SIZE, 10], name="y")

        x_image = tf.reshape(x, [const.BATCH_SIZE, 28, 28, 1])
        x_var = cnn.image_variable([const.BATCH_SIZE, 28, 28, 1], name="x_var")  # Build variable for image alteration
        print("Finished building inputs and outputs")

        # Classifier architecture (the same with that of MNIST classifier)
        # with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
        #
        #     # Convolutional layer 1
        #     W_conv1 = cnn.weight_variable([5, 5, 1, 32], name="W_conv1")
        #     b_conv1 = cnn.bias_variable([32], name="b_conv1")
        #
        #     h_conv1 = tf.nn.relu(cnn.conv2d_basic(x_var, W_conv1, b_conv1), name="h_conv1")
        #     h_pool1 = cnn.max_pool_2x2(h_conv1)
        #
        #     # Convolutional layer 2
        #     W_conv2 = cnn.weight_variable([5, 5, 32, 64], name="W_conv2")
        #     b_conv2 = cnn.bias_variable([64], name="b_conv2")
        #
        #     h_conv2 = tf.nn.relu(cnn.conv2d_basic(h_pool1, W_conv2, b_conv2), name="h_conv2")
        #     h_pool2 = cnn.max_pool_2x2(h_conv2)
        #
        #     # Fully connected layer 1
        #     h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        #
        #     W_fc1 = cnn.weight_variable([7 * 7 * 64, 1024], name="W_fc1")
        #     b_fc1 = cnn.bias_variable([1024], name="b_fc1")
        #
        #     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")
        #
        #     # Fully connected layer 2 (Output layer)
        #     W_fc2 = cnn.weight_variable([1024, 10], name="W_fc2")
        #     b_fc2 = cnn.bias_variable([10], name="b_fc2")
        #
        #     y_ = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='y_')
        y_ = mnist_classifier(x_var, 1)
        print("Finished building image generator")

        # Loss: cross entropy
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]), name="cross_entropy")
        regularization = tf.nn.l2_loss(tf.subtract(x_image, x_var), name="regularization")
        loss = cross_entropy + 0.005 * regularization

        # Gradient of loss w.r.t. input image
        # d_x = tf.gradients(cross_entropy, x_var, stop_gradients=[x_var])
        d_x = tf.gradients(loss, x_var, stop_gradients=[x_var])

        # Evaluation metrics
        prediction = tf.argmax(input=y_, axis=1)
        probability = tf.reduce_max(y_, axis=1)
        correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        print("Finished building metrics")

        # GPU memory tricks, so that computation doesn't take all GPU resource at once
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Restore model
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=["x_var"])

            model_dir = "./../classifier/model"
            saver = tf.train.Saver(variables_to_restore)
            ckpt = tf.train.get_checkpoint_state(model_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            print("Finished restoring model")

            # == Constructing adversary images == ##
            print("Start training:\n")

            ims = get_vl_samples_with_class_c(c=const.ORIG_LABEL, max_n=const.BATCH_SIZE)
            lbs = tf.one_hot(indices=tf.cast([const.TARGET_LABEL] * ims.shape[0], tf.int32), depth=10).eval()
            rate = tf.constant(const.L_RATE, dtype=tf.float32)

            sess.run(tf.assign(x_var, x_image), feed_dict={x: ims})

            # Prediction before alteration
            loss_out, entropy_out, accuracy_out, pred_out, prob_out = \
                sess.run([loss, cross_entropy, accuracy, prediction, probability], feed_dict={x: ims, y: lbs})
            print("Step = 0, loss = {}, entropy = {}, accuracy = {}".format(loss_out, entropy_out, accuracy_out))
            print("Predictions from MNIST classifier:")
            print(pred_out)
            print("Probability from MNIST classifier:")
            print(prob_out)
            print()

            delta_x = None
            step = 0
            while step < const.MAX_STEP and accuracy_out < 0.99:

                # Calculate the gradient w.r.t. input and update input images
                dx = sess.run(rate * tf.squeeze(d_x, [0]), feed_dict={x: ims, y: lbs})
                sess.run(tf.assign(x_var, tf.subtract(x_var, rate * tf.convert_to_tensor(dx, dtype=tf.float32))))

                if delta_x is None:
                    delta_x = -dx
                else:
                    delta_x -= dx

                # Predication after step steps (monitoring)
                loss_out, entropy_out, accuracy_out, pred_out, prob_out = \
                    sess.run([loss, cross_entropy, accuracy, prediction, probability], feed_dict={x: ims, y: lbs})
                print("Step = {}, loss = {}, entropy = {}, accuracy = {}".format(step + 1, loss_out, entropy_out,
                                                                                 accuracy_out))
                print("Predictions from MNIST classifier:")
                print(pred_out)
                print("Probability from MNIST classifier:")
                print(prob_out)
                print()

                step += 1

            print("Finished training\n")

            ims_alt = sess.run(x_var)
            result = format_output(ims.reshape([ims.shape[0], 28, 28, 1]), delta_x, ims_alt)

            # Save results
            save_adversarial_images(result)

            # Visualize orig images, delta modifiers, and adversarial images
            visualize_result(result)


if __name__ == "__main__":
    tf.app.run()
