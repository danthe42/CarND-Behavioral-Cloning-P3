import os
import argparse
from datetime import datetime
import tensorflow as tf
import cv2
import sklearn
import random
import numpy as np
from sklearn.utils import shuffle
from math import ceil
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.backend.tensorflow_backend import set_session
from PIL import Image
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
from scipy import ndimage

# --------------------------------------------------------------------------
# ---- parameters & constants:

crop_top, crop_bottom = 50, 20
crop_left, crop_right = 0, 0
screen_width = 320
screen_height = 160
straightdropprobability = 0.10          # drop this many percent of images with steering==0 from csv. We want to learn from action
toolowspeedlimit = 6.                   # drop the camera images below this speed (the car is too slow, steeering value is not too useful)
steering_correction = 0.2               # steering correction value for the side cameras
distortedimageaddprobability = 0.35     # parameter, should be tuned
num_epochs = 12
dropout_rate = 0.2
batch_size = 128  # Set our batch size

# --------------------------------------------------------------------------
#
class DataSet:
    def __init__(self, id):
        self.steerings = []
        self.images = []
        self.id = id

    # load image referenced by a field in the CSV file
    def _prepare_image(self, filename):
        im = Image.open(filename)
        assert (im.mode == 'RGB')  # keras preprocess expects array in RGB order
        camera_image = np.asarray(im)  # load RGB image as drive.py is also using this format.
        camera_image = camera_image[crop_top:screen_height - crop_bottom, :, :]
        return camera_image

    # add the image to our internal list
    def _reallyaddimage(self, img, steering):
        # normalization
        self.images.append((img / 127.5) - 1.0)
        self.steerings.append(steering)

    # generate distorted image randomly
    def _add_distorted_image_randomly( self, img, steering ):
        if random.random()<distortedimageaddprobability:
            distorttype = random.randint(0, 2)
            if distorttype == 0:
                img2 = ndimage.rotate(img, np.random.uniform(-3.0, 3.0), reshape=False)
                img2 = np.clip(img2, 0, 255)

            elif distorttype == 1:
                # random +- 50% brightness
                img2 = cv2.addWeighted(img, (1 + (random.random() - 0.5) / 2), img, 0.0, 0.0)
                img2 = np.clip(img2, 0, 255)

            elif distorttype == 2:
                # random +- 25% brightness
                img2 = cv2.addWeighted(img, (1 + (random.random() - 0.5) / 2), img, 0.0, 0.0)
                img2 = np.clip(img2, 0, 255)

            self._reallyaddimage( img2, steering )

    # add image to out data, and randomly an other one, generated/distorted from it
    def _addimage( self, img, steering ):
        # add image, and random generated image(es) with a predefined probability
        self._reallyaddimage(img, steering)
        self._add_distorted_image_randomly(img, steering)

    # process a line from the csv file.
    def load(self, linevec):
        for line in linevec:
            steering = float(line[3])
            speed = float(line[6])
            if speed < toolowspeedlimit:
                continue
            # centerimage
            # if steering == 0 and we would use the center camera image: drop it with "straightdropprobability" chance
            if (steering != 0) or (random.random() > straightdropprobability):
                self._addimage( self._prepare_image(line[0]), steering )
            # left image
            # create adjusted steering measurements for the side camera images if necessary
            self._addimage(self._prepare_image(line[1]), min( 1.0, steering + steering_correction ))

            # right image
            # create adjusted steering measurements for the side camera images if necessary
            self._addimage(self._prepare_image(line[2]), max( -1.0, steering - steering_correction))
        print("{} loaded and generated {} samples.".format(id, self.getsize()))

    def getsize(self):
        return 2*len(self.steerings)

    # get a random sample image with the related steering wheel angle value.
    def getrandomdata(self):
        samplelen = len(self.images)
        idx = random.randint(0, 2*samplelen - 1)

        if idx>=samplelen:
            flipped = np.fliplr(self.images[ idx-samplelen ])
            return ( -self.steerings[ idx-samplelen ], flipped )
        else:
            return ( self.steerings[ idx ], self.images[ idx ] )

class DataSets():
    def __init__(self):
        self.train_data = DataSet('Train set')
        self.validation_data = DataSet('Validation set')
        self.samples = []

    def _load_post(self, validation_set_ratio = 0.1):
            # split samples to 2 parts: train set and validation set
            train_samples, validation_samples = train_test_split(self.samples, test_size=0.15)
            self.train_data.load(train_samples)
            self.validation_data.load(validation_samples)

    # read the input data
    # If the argument is a directory path, it's content will be loaded
    # If the argument is a text file which contains a list of directories, we will load images randomly from each
    def load(self, input_data_place, load_probability):
        isfolder = os.path.isdir(input_data_place)
        if isfolder:
            self.load_lines( input_data_place, load_probability )
        else:
            with open(input_data_place) as f:
                datadirs = f.read().splitlines()
                for i in datadirs:
                    self.load_lines( i, load_probability )
        self._load_post()

    def load_lines(self, datadir, load_probability, validation_set_ratio = 0.1):
        with open(datadir + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            # read driving_log CSV file
            for line in reader:
                if random.random() < load_probability:
                    self.samples.append(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Trainer')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='model2.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='./sampledata',
        help='Either a path to the image folder, or a text file filled with images folders. This is where the train images are loaded from.'
    )
    parser.add_argument(
        'load_probability',
        type=float,
        nargs='?',
        default=1,
        help='The probability of using a given line in the input CSV file(s). Default value is 1, meaning 100%, use all input.'
    )
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True      # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)                       # set this TensorFlow session as the default session for Keras

    g_data = DataSets()
    g_data.load(args.image_folder, args.load_probability)

    def get_model(modelname):
        try:
            f = os.open(modelname, os.O_RDONLY)
            os.close(f)
            print("Loading previusly trained model.")
            return load_model( modelname )
        except:
            # create new model
            x = Sequential()

            #conv1
            x.add(Conv2D(filters=32, input_shape=(screen_height-(crop_top+crop_bottom), screen_width, 3), kernel_size=(7, 7), strides=(4, 4), padding='same', activation='tanh'))
            x.add(BatchNormalization())
            x.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            # 2nd Convolutional Layer
            x.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'))
            x.add(BatchNormalization())
            x.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            # 3rd Convolutional Layer
            x.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh'))
            x.add(BatchNormalization())

            # Passing it to a Fully Connected layer
            x.add(Flatten())
            # 1st FC
            x.add(Dense(256, activation='relu'))
            x.add(BatchNormalization())
            x.add(Dropout(0.25))

            # 2nd FC
            x.add(Dense(128, activation='relu'))
            x.add(BatchNormalization())
            x.add(Dropout(0.25))

            # Output: 1 unit, fully connected with the previous layer's output values
            x.add(Dense(1, activation='tanh'))

            x.compile(loss='mse', optimizer='adam')
            return x

    def generator(d, batch_size=32):
        num_samples = d.getsize()
        while 1:            # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                images = []
                angles = []
                for idx in range(0, batch_size):
                    ( steering_dir, image ) = d.getrandomdata()
                    images.append(image)
                    angles.append(steering_dir)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(g_data.train_data, batch_size=batch_size)
    validation_generator = generator(g_data.validation_data, batch_size=batch_size)

    conv_model = get_model(args.model)
    conv_model.summary()

    hist = conv_model.fit_generator(train_generator,
                        steps_per_epoch=ceil(g_data.train_data.getsize()/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(g_data.validation_data.getsize()/batch_size),
                        epochs=num_epochs,
                        verbose=1)
    conv_model.save(args.model)

    ### plot the training and validation loss for each epoch
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    plt.savefig('train_history_{}.png'.format(timestamp), bbox_inches='tight')

