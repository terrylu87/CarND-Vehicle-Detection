## Writeup
### Writeup

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
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/HOG_car_0.png
[image4]: ./output_images/HOG_notcar_0.png
[image5]: ./output_images/windows_64.png
[image6]: ./output_images/search_64.png
[image7]: ./output_images/heat_64.png
[image8]: ./output_images/label_64.png
[image9]: ./output_images/bbox_64.png
[video1]: ./output_videos/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `lesson_functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried 'RGB', 'HSV' and 'YCrCb' color space. The testing result shows the 'YCrCb' produced the best accuracy.
I also tried to use one channel and 'all' channels hog features. The one channel result in 95.4% test accuracy. And the all 3 channel features get over 98% accuracy.

I tuned parameters like the orientations and pixels_per_cell, but the accuracy doesn't changed that much.  

So my final parameter is following:  

color_space = 'YCrCb'  

orient = 9  # HOG orientations  

pix_per_cell = 8 # HOG pixels per cell  

cell_per_block = 2 # HOG cells per block  

hog_channel = 'ALL'  

spatial_size = (16, 16) # Spatial binning dimensions  

hist_bins = 16    # Number of histogram bins  

spatial_feat = True # Spatial features on  

hist_feat = True # Histogram features on  

hog_feat = True # HOG features on


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use HOG feature along with Spatial and Histogram features. The default choice is linear SVM. And it already provided over 98.4% classification accuracy.  

The first reviewer suggested try more parameters to get 99% accuracy. I found it's not easy to do so by using liner svc. I tried rbf kernel can produce 99.04% test accuracy but is much slower than liner kernel. I trained another SGD classifier which achieve 98.5% accuracy, but still can not get over 99%. I then tried XGBoost. After 70 rounds of training , the classifier get 99.1% test accuracy. The XGBoost classifier is almost 2 times slower than the liner model. So I decide to use liner svc for the next step.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use sliding window search taught in class to see if the parameter works for test imgages. And use the Hog sub-sampling window search discussed in the class in the video detection to save some computation cost. The parameter 'scale' can scale the size of sub-sampling windows. I tried mutiple scales, like 1, 1.25, 1.5, 1.75, 2, 3, 4. I found 1, 1.5 and 2 produced robust result.
Here is the example boxes of scale 1, 64*64
![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I uploaded a wrong video in the first submition. A video of my first attempt, with many many many false positives. That's the main reason I had to resubmit the project. I doubble check the result this time.  

Here's a [link to my video result](./output_videos/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Instead of record hot windows, I directly recorded the result heatmap in the last 10 frames of the video.  I then sum the heatmap of those frames, thresholded that summed heat map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from one frame, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on this frame of video:

### Here are one frames and it's corresponding heatmaps:

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from this frame:
![alt text][image8]

### Here the resulting bounding boxes are drawn onto the frame:
![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The classification part is raletively easy, I tried 3 sets of parameters to get 98.4% accuracy. But when I use this classifer in the test image, I found it still create many false positives. So I carefully tunning the parameters to get a resonable result on test images.  

But since those parameter is tunned for the specific test images. They do not work well on the continues video stream. The result is not stable throughout the project video, some false bounding box will come from nowhere. So I use the history information between frames to get a smoother result.   

Tunning the parameters is not a fun job, the detector is very slow on my computer. So I take 6 clips of the project video, each of them represent a typical situation.  I tunned the threshold of the heatmap to get a balance between detection and false positives. When I get a nice result on those test video clips, I apply the detector on the whole project video. Some undesirable bounding box still pop up in some frames. I then tunned the threshold for those frames and get a raletively clean result. I run tunning-testing circle for at least 20 times to get a resonable result.  

This painful searching process make me miss the yolov2, which I used for a detection project last year. The end-to-end network will do all the dirty job for you, and gives a much more promising result than the hand designed features. I'll try it in the future.
