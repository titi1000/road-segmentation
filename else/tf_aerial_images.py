"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich

This was last tested with TensorFlow 1.13.2, which is not completely up to date.
To 'downgrade': pip install --upgrade tensorflow==1.13.2
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 2
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 1
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

TEST_SIZE = 50

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

tf.app.flags.DEFINE_string(
    "train_dir",
    "C:/Users/qchap/OneDrive/Donnees/EPFL/MA1/ML/projects/project2/project_road_segmentation/tmp/segment_aerial_images",
    """Directory where to write event logs """ """and checkpoint.""",
)
FLAGS = tf.app.flags.FLAGS


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return numpy.asarray(data)

def extract_data_test(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = f"test_{i}/test_{i}"
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    return imgs


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = numpy.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = numpy.asarray(
        [value_to_class(numpy.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0
        * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
        / predictions.shape[0]
    )


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + " " + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + " " + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j : j + w, i : i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument
    data_dir = "C:/Users/qchap/OneDrive/Donnees/EPFL/MA1/ML/projects/project2/project_road_segmentation/training/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"
    test_data_filename = "C:/Users/qchap/OneDrive/Donnees/EPFL/MA1/ML/projects/project2/project_road_segmentation/test_set_images/"

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    test_data = extract_data_test(test_data_filename, TEST_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32, shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)
    )
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal(
            [5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED  # 5x5 filter, depth 32.
        )
    )
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED)
    )
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
            stddev=0.1,
            seed=SEED,
        )
    )
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED)
    )
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value * PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(unet_model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(
            img.shape[0],
            img.shape[1],
            IMG_PATCH_SIZE,
            IMG_PATCH_SIZE,
            output_prediction,
        )

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def unet_model(data, train=False, dropout_rate=0.4):
        """U-Net Model with Batch Normalization and Dropout."""
        # Encoder Path
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")(data)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)  # Add Batch Normalization
        conv1 = tf.keras.layers.Activation("relu")(conv1)  # Activation after normalization
        if train:
            conv1 = tf.keras.layers.Dropout(rate=dropout_rate)(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")(pool1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation("relu")(conv2)
        if train:
            conv2 = tf.keras.layers.Dropout(rate=dropout_rate)(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = tf.keras.layers.Conv2D(256, kernel_size=3, padding="same")(pool2)
        # conv3 = tf.keras.layers.BatchNormalization()(conv3)
        # conv3 = tf.keras.layers.Activation("relu")(conv3)
        # if train:
        #     conv3 = tf.keras.layers.Dropout(rate=dropout_rate)(conv3)
        # pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = tf.keras.layers.Conv2D(512, kernel_size=3, padding="same")(pool3)
        # conv4 = tf.keras.layers.BatchNormalization()(conv4)
        # conv4 = tf.keras.layers.Activation("relu")(conv4)
        # if train:
        #     conv4 = tf.keras.layers.Dropout(rate=dropout_rate)(conv4)
        # pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottleneck
        bottleneck = tf.keras.layers.Conv2D(256, kernel_size=3, padding="same")(pool2)
        bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
        bottleneck = tf.keras.layers.Activation("relu")(bottleneck)
        if train:
            bottleneck = tf.keras.layers.Dropout(rate=dropout_rate)(bottleneck)

        # Decoder Path
        # upsample1 = tf.keras.layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same")(bottleneck)
        # concat1 = tf.concat([upsample1, conv4], axis=-1)  # Skip connection
        # deconv1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding="same")(concat1)
        # deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
        # deconv1 = tf.keras.layers.Activation("relu")(deconv1)
        # if train:
        #     deconv1 = tf.keras.layers.Dropout(rate=dropout_rate)(deconv1)

        # upsample2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(deconv1)
        # concat2 = tf.concat([upsample2, conv3], axis=-1)  # Skip connection
        # deconv2 = tf.keras.layers.Conv2D(256, kernel_size=3, padding="same")(concat2)
        # deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
        # deconv2 = tf.keras.layers.Activation("relu")(deconv2)
        # if train:
        #     deconv2 = tf.keras.layers.Dropout(rate=dropout_rate)(deconv2)

        upsample3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(bottleneck)
        concat3 = tf.concat([upsample3, conv2], axis=-1)  # Skip connection
        deconv3 = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")(concat3)
        deconv3 = tf.keras.layers.BatchNormalization()(deconv3)
        deconv3 = tf.keras.layers.Activation("relu")(deconv3)
        if train:
            deconv3 = tf.keras.layers.Dropout(rate=dropout_rate)(deconv3)

        upsample4 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(deconv3)
        concat4 = tf.concat([upsample4, conv1], axis=-1)  # Skip connection
        deconv4 = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")(concat4)
        deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
        deconv4 = tf.keras.layers.Activation("relu")(deconv4)
        if train:
            deconv4 = tf.keras.layers.Dropout(rate=dropout_rate)(deconv4)

        # Flatten the output to get (batch_size, height * width * channels)
        flattened = tf.keras.layers.Flatten()(deconv4)

        # Dense layer to reduce to (batch_size, NUM_LABELS) for final classification output
        output = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")(flattened)  # sigmoid for single-class classification

        # Return the output tensor
        return output

    def unet_model2(data, train=False, dropout_rate=0.5):
        """U-Net Model with Dropout."""
        
        # Encoder Path
        conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        if train:
            relu1 = tf.nn.dropout(relu1, rate=dropout_rate)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        if train:
            relu2 = tf.nn.dropout(relu2, rate=dropout_rate)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Bottleneck
        bottleneck_conv = tf.nn.conv2d(pool2, bottleneck_weights, strides=[1, 1, 1, 1], padding="SAME")
        bottleneck_relu = tf.nn.relu(tf.nn.bias_add(bottleneck_conv, bottleneck_biases))
        if train:
            bottleneck_relu = tf.nn.dropout(bottleneck_relu, rate=dropout_rate)

        # Decoder Path (Upsampling with skip connections)
        output_shape = [data.shape[0], pool1.shape[1], pool1.shape[2], upconv1_weights.shape[-1]]
        output_shape = [dim.value if hasattr(dim, 'value') else dim for dim in output_shape]
        upsample1 = tf.nn.conv2d_transpose(
            bottleneck_relu, upconv1_weights,
            output_shape=output_shape,
            strides=[1, 2, 2, 1], padding="SAME"
        )
        concat1 = tf.concat([upsample1, relu2], axis=-1)  # Skip connection
        deconv1 = tf.nn.conv2d(concat1, deconv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu3 = tf.nn.relu(tf.nn.bias_add(deconv1, deconv1_biases))
        if train:
            relu3 = tf.nn.dropout(relu3, rate=dropout_rate)

        output_shape = [data.shape[0], data.shape[1], data.shape[2], upconv2_weights.shape[-1]]
        output_shape = [dim.value if hasattr(dim, 'value') else dim for dim in output_shape]
        upsample2 = tf.nn.conv2d_transpose(
            relu3, upconv2_weights,
            output_shape=output_shape,
            strides=[1, 2, 2, 1], padding="SAME"
        )
        concat2 = tf.concat([upsample2, relu1], axis=-1)  # Skip connection
        deconv2 = tf.nn.conv2d(concat2, deconv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu4 = tf.nn.relu(tf.nn.bias_add(deconv2, deconv2_biases))
        if train:
            relu4 = tf.nn.dropout(relu4, rate=dropout_rate)

        # Flatten the spatial dimensions into a vector for classification
        flattened = tf.reshape(relu4, [data.shape[0], -1])  # Flatten to (batch_size, num_features)
        
        # Fully connected layer to reduce to 2 classes (no spatial dimensions)
        fc1 = tf.matmul(flattened, fc1_weights) + fc1_biases
        fc1_relu = tf.nn.relu(fc1)
        if train:
            fc1_relu = tf.nn.dropout(fc1_relu, rate=dropout_rate)
        
        # Output logits (shape: batch_size x 2)
        logits = tf.matmul(fc1_relu, fc2_weights) + fc2_biases
        
        if train:
            summary_id = "_0"
            s_data = get_image_summary(data)
            tf.summary.image("summary_data" + summary_id, s_data, max_outputs=3)
            tf.summary.image("summary_output" + summary_id, logits, max_outputs=3)

        return logits

    # Training computation: logits + cross-entropy loss.
    logits = unet_model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
    # print('logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape()))

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=train_labels_node, logits=logits
        )
    )

    tf.summary.scalar("loss", loss)

    all_params_node = [
        conv1_weights,
        conv1_biases,
        conv2_weights,
        conv2_biases,
        fc1_weights,
        fc1_biases,
        fc2_weights,
        fc2_biases,
    ]
    all_params_names = [
        "conv1_weights",
        "conv1_biases",
        "conv2_weights",
        "conv2_biases",
        "fc1_weights",
        "fc1_biases",
        "fc2_weights",
        "fc2_biases",
    ]
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (
        tf.nn.l2_loss(fc1_weights)
        + tf.nn.l2_loss(fc1_biases)
        + tf.nn.l2_loss(fc2_weights)
        + tf.nn.l2_loss(fc2_biases)
    )
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.00001,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True,
    )
    # tf.scalar_summary('learning_rate', learning_rate)
    tf.summary.scalar("learning_rate", learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, global_step=batch
    )

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(unet_model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=s.graph)

            print("Initialized!")
            # Loop through training steps.
            print(
                "Total number of iterations = "
                + str(int(num_epochs * train_size / BATCH_SIZE))
            )

            training_indices = range(train_size)

            losses = []
            for iepoch in range(num_epochs):
                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                steps_per_epoch = int(train_size / BATCH_SIZE)

                loss_plot = 0
                for step in range(steps_per_epoch):
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset : (offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {
                        train_data_node: batch_data,
                        train_labels_node: batch_labels,
                    }

                    if step == 0:
                        summary_str, _, l, lr, predictions = s.run(
                            [
                                summary_op,
                                optimizer,
                                loss,
                                learning_rate,
                                train_prediction,
                            ],
                            feed_dict=feed_dict,
                        )
                        summary_writer.add_summary(
                            summary_str, iepoch * steps_per_epoch
                        )
                        summary_writer.flush()

                        print("Epoch %d" % iepoch)
                        print("Minibatch loss: %.3f, learning rate: %.6f" % (l, lr))
                        print(
                            "Minibatch error: %.1f%%"
                            % error_rate(predictions, batch_labels)
                        )

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict,
                        )
                    
                    loss_plot += l
                loss_plot /= steps_per_epoch
                print(f"Loss of epoch {iepoch}: {loss_plot}")

                losses.append(loss_plot)
                plt.plot(losses)
                plt.savefig("plot.png")

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        print("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, TRAINING_SIZE + 1):
            pimg = get_prediction_with_groundtruth(train_data_filename, i)
            Image.fromarray(pimg).save(
                prediction_training_dir + "prediction_" + str(i) + ".png"
            )
            oimg = get_prediction_with_overlay(train_data_filename, i)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        
        # predict on test set
        print("Running prediction on test set")
        prediction_test_dir = "predictions_test/"
        if not os.path.isdir(prediction_test_dir):
            os.mkdir(prediction_test_dir)
        for i in range(1, TEST_SIZE + 1):
            pred = get_prediction(test_data[i-1])
            pred = (pred * 255).astype(numpy.uint8)
            Image.fromarray(pred).save(prediction_test_dir + "pred" + str(i) + ".png")


if __name__ == "__main__":
    tf.app.run()
