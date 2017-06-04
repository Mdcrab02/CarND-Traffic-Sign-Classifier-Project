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

Currently the model has an accuracy of 87% on the training set, and an accuracy of 95% on the validation set.

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

* The size of the new training set is: 47,429
* The portion of the training set used for validation is: 20%
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
is difficult.  When I tried to convert the images to grayscale (for further processing), they always ended up being teal and yellow.

I was able to successfully process the images from RGB into YUV, but the results were that
the model's performance was really poor.

I wanted to convert the images into grayscale and use the adaptive histogram method to equalize them, but could not get anything to convert them to grayscale.  Errors.  Errors everywhere.

To generate more data I combined the training and test sets into one large training set.

### Designing the Model

I designed the model in Keras based off some of the model architectures developed by Google.

The model begins with two convolutional layers that use max pooling, batch normalization, and a 'relu' activation.  I started with relu, then went to sigmoid, and then back to relu.  I then added a flatted layer before passing the outputs into dense nodes with 128, 256, and 128 units respectively.  All used dropout and a relu activation.  Then I added another
layer with Dense() for the final layer associated with the number of classes before
passing the output to a softmax regression activation layer to get my logits.

### Performance Improvements

In addition to my model architecture changes, I also added some Keras code to stop training the model once it reached an acceptable minimum validation loss after sets of 2 epochs.

The model could be improved further with more images and by making it deeper.  However, I think 87% training accuracy and 95% validation accuracy is pretty damn good.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Model Architecture Table

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution (32x3x3)     	| Convolutional layer 	|
| Activation	    | ReLu activation      									|
| Max Pooling (2,2)					|	2x2 Max Pooling Layer											|
| Normalization					|	Batch Normalization Layer											|
| Convolution (64x3x3)     	| Convolutional layer 	|
| Activation	    | ReLu activation      									|
| Normalization					|	Batch Normalization Layer											|
| Max Pooling (2,2)					|	2x2 Max Pooling Layer											|
| Flatten		| Flatten Layer        									|
| Dense(128)				| Regular densely-connected layer, 128 units        									|
|	Activation					|	ReLu activation											|
| Dropout      	| 50% chance for dropout 				|
| Dense(256)				| Regular densely-connected layer, 256 units        									|
|	Activation					|	ReLu activation											|
| Dropout      	| 50% chance for dropout 				|
| Dense(128)				| Regular densely-connected layer, 128 units        									|
|	Activation					|	ReLu activation											|
| Dropout      	| 50% chance for dropout 				|
|	Dense(43)					|	Regular densely-connected layer, 43 units											|
|	Activation					|	Final activation - Softmax regression											|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

### Model Training

To train the model, I used the model.fit() function from Keras after compiling it with
Keras.  Eventually, I used the 'adam' optimizer instead of the others, like SGD, because
the performance differences were not a big deal.  I mostly added the other optimizers to the
code to check because initially the performance was very poor.

The model fit used the raw images (see preprocessing problems above), and a one-hot encoding
vector generated using scikit-learn's LabelBinarizer().  There are a few different ways I could
have done the one-hot encoding, but using scikit is easier.

After setting the parameters for my callbacks in Keras, the training stopped at 4 epochs because the validation loss had ceased to have a smooth curve down to the minimum after 2 epochs.  It went down, then up, then down again.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

### Model Performance

My final model results were:

* training set accuracy of: 87%
* validation set accuracy of: 95%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

  I started off with regular TensorFlow code because that's what was covered prior to the assignment.  The architecture was really basic and did not include dropout or any kind
  of pooling.

* What were some problems with the initial architecture?

  The code quickly became a bit of a mess, and the performance was really bad.  The accuracy
  started out around 0.05%.  The validation accuracy also hovered around 0%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  At first the model was very underfit.  While image processing could help, the main issues were that the model was not deep enough and it did not have enough images to learn from.

* Which parameters were tuned? How were they adjusted and why?

  Almost everything, because the performance of the model was poor.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  Convolutional layers help scan over the images in layers based on the number of convolutions, kernel size, and stride.  Using two of these helped my model deal with the complex color images.  Dropout on my dense layers also regularized them to mitigate overfitting issues.

### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

I Google searched for German traffic signs based on the sign names (labels) from the data sets.

After wading through a sea of images covered with watermarks, I finally found some to use for testing.  What kind of person takes a camera to a publicly funded and publicly displayed traffic sign, and slaps a watermark on it before putting it on the internet for people to use?  It's a traffic sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop      		| Stop   									|
| Roundabout mandatory     			| Keep right 										|
| No entry					| No entry											|
| Road work	      		| Road work					 				|
| Right turn ahead			| Right turn ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.315), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .315         			| Stop sign   									|
| .091     				| U-turn 										|
| .08					| Yield											|
| .075	      			| Bumpy Road					 				|
| .075				    | Slippery Road      							|
