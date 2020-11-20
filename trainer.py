import os
import argparse
import tensorflow as tf
import csv
import cv2
import sklearn
import random
import numpy as np
from sklearn.utils import shuffle
from math import ceil
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, GlobalAveragePooling2D, Dropout, Input
from keras.backend.tensorflow_backend import set_session
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Trainer')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='model.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='../sampledata/',
        help='Path to image folder. This is where the train images are loaded from.'
    )
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True      # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)                       # set this TensorFlow session as the default session for Keras

    # ---------------------------------------------------------------------------------------
    datadir = args.image_folder
    model_filename = args.model
    num_epochs = 3
    dropout_rate = 0.2
    steering_correction = 0.1                                       # steering correction value for the side cameras
    batch_size=64                                                   # Set our batch size
    crop_top, crop_bottom = 60, 20
    crop_left, crop_right = 10, 11
    screen_width = 320
    screen_height = 160
    ch, row, col = 3, screen_height - (crop_top+crop_bottom), 320             # Trimmed image format
    # ---------------------------------------------------------------------------------------

    # read driving_log CSV file
    samples = []
    with open(datadir+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        firstline = next(reader)
        for line in reader:
            samples.append(line)

    # split samples to 2 parts: train set and validation set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    global imageid, exportfirstbatchimages
    imageid = 0

    # should the forst batch images be saved ?
    exportfirstbatchimages = False

    def get_model(input_tensor):
        try:
            f = os.open(model_filename, os.O_RDONLY)
            os.close(f)
            print("Loading previusly trained model.")
            return load_model( model_filename )
        except:
            print("Creating model based on the InceptionV3 network.")
            conv_model = InceptionV3(weights='imagenet',include_top=False, input_tensor=input_tensor)

            # the bottom of the network should not be trained
            totallayers = len(conv_model.layers)
            nottrainable = totallayers * 2 // 3
            print("Total number of layers in the base model: {} The bottom {} layers shall not be trained.".format(totallayers, nottrainable))
            for layeridx in range(0, nottrainable):
                conv_model.layers[layeridx].trainable = False

            x = GlobalAveragePooling2D()(conv_model.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            out = Dense(1)(x)
            model = Model(inputs=preprocess_in, outputs=out)
            model.summary()
            model.compile(loss='mse', optimizer='adam')
            return model

    def generator(samples, batch_size=32):
        global imageid, exportfirstbatchimages

        wassave = False

        num_samples = len(samples)
        while 1:            # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    i = random.randint(0,2)
                    #datadir+'/IMG/'+
                    filename = batch_sample[i].split('/')[-1]
                    im = Image.open( filename )
                    assert(im.mode == 'RGB')                # keras preprocess expects array in RGB order

                    camera_image = np.asarray(im)           # load RGB image as drive.py is also using this format.

                    # Preprocess step 1: crop the nonimportant top and bottom part of the image
                    # Preprocess step 2: centered around zero with small standard deviation
                    # So: cut/resize the image to the expected size, but keep the aspect ratio
                    # the standard preprocess_input() will handle the normalization
                    camera_image = camera_image[ crop_top:screen_height-crop_bottom, crop_left:screen_width-crop_right, :]
                    camera_image = np.pad(camera_image, ((109, 110),(0,0),(0,0)))

                    steering_dir = float(batch_sample[3])

                    # create adjusted steering measurements for the side camera images if necessary
                    if i == 1:          # left camera image
                        steering_dir = min( 1.0, steering_dir + steering_correction )
                    if i == 2:          # right camera image
                        steering_dir = max( -1.0, steering_dir - steering_correction)

                    # Randomly flip the image horizontally
                    bFlip = bool(random.getrandbits(1))
                    if bFlip is True:
                        center_image = np.fliplr(camera_image)
                        steering_dir = -steering_dir

                    camera_image_tosave = camera_image
                    camera_image = preprocess_input(camera_image)

                    if (exportfirstbatchimages is True) and (wassave is False) and (offset == 0):
                        # atomic increment would be better, but it's not so critical
                        imageid = imageid + 1
                        cv2.imwrite( "output_images/cnninput{}.jpg".format(imageid), cv2.cvtColor(camera_image_tosave, cv2.COLOR_RGB2BGR) )
                    images.append(camera_image)
                    angles.append(steering_dir)

                wassave = True
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    preprocess_in = Input( shape=(299, 299, 3) )

    conv_model = get_model(preprocess_in)
    conv_model.summary()

    # increase the number of batches because of the randomly generated distorted image samples (by a factor of 5)
    hist = conv_model.fit_generator(train_generator,
                        steps_per_epoch=ceil(5*len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(5*len(validation_samples)/batch_size),
                        epochs=num_epochs,
                        verbose=1)
    conv_model.save(model_filename)

    ### print the keys contained in the history object
    print(hist.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('output_images/train_history.png', bbox_inches='tight')
    #plt.show()
