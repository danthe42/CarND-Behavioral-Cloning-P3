import os
import tensorflow as tf
import csv
import sklearn
import random
import numpy as np
from sklearn.utils import shuffle
from math import ceil
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, GlobalAveragePooling2D, Dropout, Input
from keras.backend.tensorflow_backend import set_session
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.layers.merge import concatenate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True      # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)                       # set this TensorFlow session as the default session for Keras

# ---------------------------------------------------------------------------------------
# constants, parameters
datadir = '../sampledata'
dropout_rate = 0.2
steering_correction = 0.2                                       # steering correction value for the side cameras
batch_size=32                                                   # Set our batch size
crop_top, crop_bottom = 60, 20
ch, row, col = 3, 160 - (crop_top+crop_bottom), 320             # Trimmed image format
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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:            # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                i = random.randint(0,2)
                filename = datadir+'/IMG/'+batch_sample[i].split('/')[-1]
                center_image = np.asarray(Image.open( filename ))           # load RGB image as drive.py is also using this format.
                steering_dir = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images if necessary
                if i == 1:          # left camera image
                    steering_dir = min( 1.0, steering_dir + steering_correction )
                if i == 2:          # right camera image
                    steering_dir = max( -1.0, steering_dir - steering_correction)

                # Randomly flip the image horizontally
                bFlip = bool(random.getrandbits(1))
                if bFlip is True:
                    center_image = np.fliplr(center_image)
                    steering_dir = -steering_dir

                images.append(center_image)
                angles.append(steering_dir)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Preprocess step 1: crop the nonimportant top and bottom part of the image
# Preprocess step 2: centered around zero with small standard deviation

preprocess_in = Input( shape=(160,320,3) )
preprocess_1 = Cropping2D(cropping=((crop_top,crop_bottom), (0,0)))(preprocess_in)
preprocess_out = Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch))(preprocess_1)
#preprocess_model = Model( preprocess_in, preprocess_out )

#array_input = Input(shape=(row, col, ch))
conv_model = InceptionV3(weights='imagenet',include_top=False, input_tensor=preprocess_out)
for layer in conv_model.layers:
            layer.trainable = False

x = GlobalAveragePooling2D()(conv_model.output)

#x = GlobalAveragePooling2D()(conv_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(dropout_rate)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(dropout_rate)(x)
out = Dense(1)(x)
model = Model(inputs = preprocess_in, outputs=out)
model.summary()
#model_final = out( preprocess_model( array_input ))
#out.summary()
#conv_model.summary() # ?

#model.add(Flatten(input_shape=(row,col,3)))
#model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
hist = model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=5,
                    verbose=1)
model.save('model.h5')
