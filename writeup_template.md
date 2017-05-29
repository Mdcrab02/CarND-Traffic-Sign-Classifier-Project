# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Overview

For this project, I initially started to use base TensorFlow for the model architecture
of the convolutional neural net design, but decided to use Keras to wrap TensorFlow.  Not
only is Keras faster for me to code, it is also much simpler for me to debug when I run
into problems.  However, I still spend a long time looking at the API reference docs
for Keras when things went awry.

---

[//]: # (Image References)

[image1]: 1.png "Visualization1"
[image2]: 2.png "Visualization2"
[image3]: 3.png "Visualization3"
[image4]: ./new_images/1.png "Traffic Sign 1"
[image5]: ./new_images/2.png "Traffic Sign 2"
[image6]: ./new_images/3.png "Traffic Sign 3"
[image7]: ./new_images/4.png "Traffic Sign 4"
[image8]: ./new_images/5.png "Traffic Sign 5"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

You're reading it! Here is a link to my [project code](https://github.com/mdcrab02/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Keras.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34,799
* The size of the validation set is: 4,410
* The size of test set is: 12,630
* The shape of a traffic sign image is: 32 x 32 x 3
* The number of unique classes/labels in the data set is: 43

#### Graphical Exploration

Here is an exploratory visualization of the data set. I have bar charts for the following:

* The number of occurrences of each sign, by sign ID, in the training data

![alt text][image1]

* The number of occurrences of each sign, by sign ID, in the test data

![alt text][image2]

* The number of occurrences of each sign, by sign ID, in the validation data

![alt text][image3]

I also explored 30 random samples of images from the training data to get a better
understanding of the training set.  I live in the USA, and anticipated many unfamiliar
looking signs as the data set contains signs from Germany.

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

### Image Processing

So, the images in the data sets are interesting, to say the least.  Many of the images
in the data sets are very blurry or very dark.  Even for human eyes, interpreting them
is difficult.  I tried several different methods for preprocessing the images, but
ran into a lot of difficulty doing so.  For example, when I tried to convert the images
to grayscale (for further processing), they always ended up being teal and yellow.

I was able to successfuly process the images from RGB into YUV, but the results were that
the model's performance was really poor.

Eventually, I gave up for the sake of time and decided to teach the model from the raw
images in the data set.  In the future I would like to figure out how to better process
these images.

After reading the description for this section, I presume that some are generating more data
for training.  While this was not covered in the lecture material, I believe I know how
this could be implemented for future iterations of the process.

### Designing the Model

I designed the model in Keras based off some of the model architectures covered in class
using base TensorFlow.

The model begins with a convolutional layer that uses max pooling and dropout prior to
reaching the first activation node.  While most of the class has used 'Relu' activation
functions, I achieved better results from the sigmoid transfer function.  I then added a
flatted layer before passing the outputs into a dense node of size 128 to match the batch
size.  Once again, this goes into a sigmoid activation layer.  Then I added another
layer with Dense() for the final layer associated with the number of classes before
passing the output to a softmax regression activation layer.

### Performance Improvements

I have CUDA and TensorFlow-GPU set up on my maching to utilize the CUDA cores in my
GTX 960, but I was unable to get the environment for term1 to use 'gpu:0'.  This
prevented me from increasing the number of epochs to train the model up to 100.  My
other environments can use my GPU just fine, but if I do that for this project the code
breaks.

Specifically, 'tf.python.control_flow_ops = tf' does not work with the new versions
of Keras and/or Tensorflow.  I could not find out why.

In the future I would like to give the model more training examples and merge the
validation pickle into one pickle with the training data, because I can split off
the validation with Keras in the model.fit.  Unfortunately, I could not figure out how
to combine these pickles.  I wish we were just given a training and test set.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Model Architecture Table

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution (32x3x3)     	| Convolutional layer 	|
| Max Pooling (2,2)					|	2x2 Max Pooling Layer											|
| Dropout      	| 50% chance for dropout 				|
| Activation	    | Sigmoid activation      									|
| Flatten		| Flatten the input        									|
| Dense(128)				| Regular densely-connected layer, 128 units        									|
|	Activation					|	Sigmoid activation											|
|	Dense(43)					|	Regular densely-connected layer, 43 units											|
|	Activation					|	Final activation - softmax regression											|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

### Model Training

To train the model, I used the model.fit() function from Keras after compiling it with
Keras.  Eventually, I used the 'adam' optimizer instead of the others, like SGD, because
the performance differences were not a big deal.  I mostly added the other optimizers to the
code to check because initially the performance was very poor.

The model fit used the raw images (see preprocessing problems above), and a one-hot encoding
vector generated using scikit-learn's LabelBinarizer().  There are a few different ways I could
have done the one-hot encoding, but using scikit is easier.

I limited the training to 3 epochs for time, because I was not able to use my GPU.  Were I
able to use my GPU as planned, I would have trained on 100 epochs.

My current model overfits anyway, so I left the number of epochs at 3 until I resolve the overfitting.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

### Model Performance

My final model results were:

* training set accuracy of: 80%
* validation set accuracy of: 19%
* test set accuracy of: 80%

These numbers are from the first run of 10 epochs from the model.  After that, I noticed
that the rubric wanted us to get to 93% validation accuracy.  When I ran the code again
the validation accuracy did not get anywhere near that high.  It looks as though it varies
wildly each time the code is run.

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

  I started off with regular TensorFlow code because that's what was covered prior to the assignment.  The architecture was really basic and did not include dropout or any kind
  of pooling.

* What were some problems with the initial architecture?

  The code quickly became a bit of a mess, and the performance was really bad.  The accuracy
  started out around 0.05%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  At first the model was very underfit.  The accuracy was really low, and I believe the
  cuplrit was the image processing.  I was not able to improve the performance by changing
  the image processing, so I played around with the structure of the model until the
  performance reached acceptable levels

* Which parameters were tuned? How were they adjusted and why?

  Almost everything, because the performance of the model was poor.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  I used both a convolutional layer and dropout.  I figured dropout would help prevent
  overfitting, but the model ended up overfitting anyway.

### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

I Google searched for German traffic signs, but I suspect some of them are not German.

The blurriness, contrast, and light level in the images make them difficult to classify.  This
is especially true given that I could not get the image processing to cooperate.  I chose
some example images that did not require heavy preprocessing.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop      		| Stop   									|
| Roundabout mandatory     			| Roundabout mandatory 										|
| Parking?					| Roundabout mandatory											|
| Children crossing	      		| Children Crossing					 				|
| Right turn ahead			| Right turn ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.  This performance is much better than the validation accuracy.  It seems the model still requires
more improvements.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.315), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .315         			| Stop sign   									|
| .091     				| U-turn 										|
| .08					| Yield											|
| .075	      			| Bumpy Road					 				|
| .075				    | Slippery Road      							|
