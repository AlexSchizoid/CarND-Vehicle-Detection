
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carvsnot.jpg
[image2]: ./output_images/hog.jpg
[image3]: ./output_images/windows.jpg
[image4]: ./output_images/heatmap.jpg
[image5]: ./output_images/labels.jpg
[image6]: ./output_images/final.jpg
[image7]: ./output_images/windows_scales.jpg
[video1]: ./result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the "Prepare features". I use the get_hog_features to extract hog paraameters from a channel. Extracting the hog features from all channels of an image is a step in the extract_features function.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Besides the hog features, in the extract_features function i also obtain spatial_features and histogram features. Spatial features are actually blocks of pixels unraveled in a vector. For the histogram features I take the histogram of each channel in the YCrCb color space and flatten them in a vector.

The final configuration parameters for the extraction features were chosen through trial and error until I get a sufficiently high accuracy in the classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the data provided by udacity which consists of both vehicles and non-vechicles. I took aside 10% of the samples for a test test set and trained the SVM on the training set. I obtained a Test Accuracy of SVC =  0.9887. The relevant code is in the "Prepare features" and "Train SVM" sections in the Jupyter notebook.

The three types of features, spatial, hist and hog are concatenated together in a vector. Each image is processed and produces this kind of vector. Afterwards this is fed to the Sklearn SVM classifier.

The configuration parameter for the feature extraction functions is:
<pre>
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
</pre>

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use small, medium and large window size by varying the scale parameter given to the find_cars function from the lesson code. The hog features are only computed once, and then can be subsampled to get each overlaying window. Using smaller scales than 1.0 just adds more false positives. Here is a picture of the different scales of windows used and their positions:

![alt text][image7]

With these windows we can use the classifier to try and predict if cars are in the image. When the classifier predicts a car then we generate coordinates for boxes that enclose that prediction. Here is an example:

![alt text][image3]

False positives(like in the previous image) might be detected so in order to remove these outliers we need to implement a rejection mechanism. For each pixel inside the determined prediction we add the value 1 to a seperate heatmap image. Each time the pixel is found inside a box that the coresponding pixel in the heatmap gets "warmer". Using these method we can now select patches of the heatmap that have pixels with values above the threshold. For example here is the heatmap obtained by thresholding to 1:

![alt text][image4]

Using the label function we can now separate detected instances in the heatmap. Here is an example image:

![alt text][image5]

Finally this allows is at the end to draw a rectangle around each found label, this being the definite detection of the pipeline. Here is an example image:

![alt text][image6]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video implementation mostly follows the implementantion of the single image pipeline. In the video pipeline i decided to store the heatmaps from the last 10 frames in order to produce a smoother heatmap. This means that the boxes around the vehicles are less wobbly. The class CarDetector is used as storage instance for the detections and is passed as a parameter to the process_frame function. The features chosen seem to be working well for this video, the ammount of false positives being minimum.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I discovered that it is very important to have a balanced training set - relatively equal numbers of the two classes(car, non-car). I previously made a mistake when i built my training set and found that the pipeline detected a lot of false positives because of the class imbalances. 

The chosen procedure seems to take a long time, 1 sec per iteration so it would definitely not work as-is in a real time environment. 

I would optimize the scales for the windows chosen and their position. I believe this will allow the algorithm to run faster. Also the configuration parameters for the feature extraction could also be fiddled with to provide a faster execution with not too big of a loss in classification accuracy.
More training samples can be added to the training set further generalizing the model and improving it's accuracy.

But ultimately i believe this is still a pretty computationally intesive approach to run on a real-time feed. I suspect that a deep learning approach to identifying objects will run much faster and produce more accurate results. 
