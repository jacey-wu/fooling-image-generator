import tensorflow as tf
from time import localtime, strftime
import os
import cv2

import utils.cnn_utils as cnn
import utils.adversary_utils as adv


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(os.path.join(".", "..", "MNIST_data"), one_hot=True)


def get_vl_samples_with_class_c(c=2, max_n=10):
    ims = mnist.validation.images
    lbs = mnist.validation.labels

    idx = [sample[c] == 1 for sample in lbs]

    ims_c = ims[idx][:]
    return ims_c[0:min(max_n, ims_c.shape[0])][:]


def main(argv=None):

    with tf.device('/gpu:0'):

        # Input layer
        x = tf.placeholder(tf.float32, shape=[15, 784], name="x")
        y = tf.placeholder(tf.float32, shape=[15, 10], name="y")

        x_image = tf.reshape(x, [15, 28, 28, 1])
        x_var = adv.image_variable([15, 28, 28, 1], name="x_var")  # TODO: make size of x_var dynamic

        print("Finished building inputs and outputs")

        # Classifier architecture
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):

            # Convolutional layer 1
            W_conv1 = adv.weight_variable([5, 5, 1, 32], name="W_conv1")
            b_conv1 = adv.bias_variable([32], name="b_conv1")

            h_conv1 = tf.nn.relu(cnn.conv2d_basic(x_var, W_conv1, b_conv1), name="h_conv1")
            h_pool1 = cnn.max_pool_2x2(h_conv1)

            # Convolutional layer 2
            W_conv2 = adv.weight_variable([5, 5, 32, 64], name="W_conv2")
            b_conv2 = adv.bias_variable([64], name="b_conv2")

            h_conv2 = tf.nn.relu(cnn.conv2d_basic(h_pool1, W_conv2, b_conv2), name="h_conv2")
            h_pool2 = cnn.max_pool_2x2(h_conv2)

            # Fully connected layer 1
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

            W_fc1 = adv.weight_variable([7 * 7 * 64, 1024], name="W_fc1")
            b_fc1 = adv.bias_variable([1024], name="b_fc1")

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

            # Fully connected layer 2 (Output layer)
            W_fc2 = adv.weight_variable([1024, 10], name="W_fc2")
            b_fc2 = adv.bias_variable([10], name="b_fc2")

            y_ = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='y_')
        print("Finished building image generator")

        # Loss
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]), name="cross_entropy")

        d_x = tf.gradients(cross_entropy, x_var, stop_gradients=[x_var])

        # Evaluation metrics
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        prediction = tf.argmax(input=y_, axis=1)
        print("Finished building metrics")

        # # Construct image step
        # construct_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=[x_var])
        # print("Finished building adversary training operation")

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

            # Constructing adversary images
            cls_init = 2
            cls_end = 6

            ims = get_vl_samples_with_class_c(c=cls_init, max_n=15)
            lbs = tf.one_hot(indices=tf.cast([cls_end] * ims.shape[0], tf.int32), depth=10).eval()

            max_step = 15
            sess.run(tf.assign(x_var, x_image), feed_dict={x: ims})

            entropy_out, accuracy_out, pred_out, image_out = \
                sess.run([cross_entropy, accuracy, prediction, x_var], feed_dict={x: ims, y: lbs})
            print("Step = -1, entropy = {}, accuracy = {}".format(entropy_out, accuracy_out))
            print(pred_out)
            # cv2.imshow("Image out", image_out[0])
            # cv2.waitKeyEx(0)
            # cv2.destroyAllWindows()

            for r in [1]:
                rate = tf.constant(r, dtype=tf.float32)

                print("==================================")
                print("when update rate = ", r)

                for step in range(max_step):
                    print("X_var")
                    image_out = sess.run(x_var)
                    print(image_out[0][10][16:20])

                    sess.run(tf.assign(x_var, tf.subtract(x_var, rate * tf.squeeze(d_x, [0]))), feed_dict={x: ims, y: lbs})

                    entropy_out, accuracy_out, pred_out, image_out = \
                        sess.run([cross_entropy, accuracy, prediction, x_var], feed_dict={x: ims, y: lbs})
                    print("Step = {}, entropy = {}, accuracy = {}".format(step, entropy_out, accuracy_out))
                    print(pred_out)

                    cv2.imwrite("im_step_{}_idx_{}.jpg".format(step, 0), image_out[0])

                    cv2.imshow("Image out", image_out[0])
                    cv2.waitKeyEx(0)
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    tf.app.run()
