# **Behavioral Cloning** 

## Project Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model first preprocesses the data using a Keras lambda layer (code line 216). 

After this first layer, there comes three 2D convolution layers with 7x7, 5x5 and 3x3 filter sizes, (4,4), (2,2) and (1,1)  strides and depths:  32, 64 and 128. (model.py lines 213-247) 

After flattening the 3rd convolutional layer's output, 3 fully connected layers are following with sizes: 256,128, and 1. The network has only one output value, the output of the last TANH activation function in the interval ]-1, 1[ and that is the steering wheel's desired angle. 

The model includes RELU layers as the activation function after the wide fully connected layers, and TANH activation after the convolutional, and the last fully connected layers. 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 237, and 242).  Moreover, I'm using Keras's BatchNormalization class on the outputs of the internal layers which uses a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 247).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road,  and collected data on both tracks, and in each direction. 

I'm also using generated (distorted) images for training, based on the originals. For that I'm using a random brightness modification (with a random factor between -50% and +50%), or a random rotation by a random value between -3 degrees and +3 degrees.  

And I'm using an other, simpler, but very useful data generation: horizontal flipping. Just by mirroring the right and left direction I effectively created 100% more data.    

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was:

- I started with the LeNet architecture first, which is an excellent network for image recognition tasks, 
- Then added a special Lambda layer before its first layer for normalizing the input data, preferably on GPU.
- Then set the output layer to a fully connected layer with only 1 unit, because I have only 1 output value.    
- I put in 2 dropout layers between the fully connected layers for more robustness, and prevent overfitting.
- Also I have inserted BatchNormalization layers in the network to keep the mean and standard deviation of the parameters.  
- I changed a few layer activations from RELU to TANH, to limit the output values between -1 and 1.
- I also tuned the parameters of the convolution layers, I choose the best model after training with different parameter combinations.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or took a  wrong maneuver. To improve the driving behavior in these cases, I recorded short clips around that area taking that specific maneuver multiple times, then trained the network with it.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 213-247) consisted of a convolution neural network with the following layers and layer sizes:

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            			     (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            				 (None, 23, 80, 32)        4736
_________________________________________________________________
batch_normalization_1(Batch)  		   (None, 23, 80, 32)        128
_________________________________________________________________
max_pooling2d_1(MaxPooling2) 		(None, 12, 40, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            				 (None, 6, 20, 64)         51264
_________________________________________________________________
batch_normalization_2(Batch) 			(None, 6, 20, 64)         256
_________________________________________________________________
max_pooling2d_2(MaxPooling2) 		 (None, 3, 10, 64)         0
_________________________________________________________________
conv2d_3 (Conv2D)            				 (None, 3, 10, 128)        73856
_________________________________________________________________
batch_normalization_3 					    (Batch (None, 3, 10, 128)        512
_________________________________________________________________
flatten_1 (Flatten)          					   (None, 3840)              0
_________________________________________________________________
dense_1(Dense)             					 (None, 256)               983296
_________________________________________________________________
batch_normalization_4 (Batch) 		   (None, 256)               1024
_________________________________________________________________
dropout_1 (Dropout)          				   (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              					(None, 128)               32896
_________________________________________________________________
batch_normalization_5(Batch) 			(None, 128)               512
_________________________________________________________________
dropout_2 (Dropout)          				  (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              				  (None, 1)                 129

Total params: 1,148,609
Trainable params: 1,147,393
Non-trainable params: 1,216



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
