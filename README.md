# FasterRCNN_finetuned
Faster R-CNN is an object detection model that extracts features using a pre-trained CNN. Read more https://arxiv.org/abs/1506.01497.

A repository created by Adonais Ray Maclang. Here, the Faster R-CNN model is finetuned using the drinks dataset containing 3 classes: water, coke, juice.

To train, unzip the drinks dataset into the main folder (https://bit.ly/adl2-ssd) and run train.py. To see results on test_split, run test.py. Alternatively, view the Google Colab Notebooks. 

See releases for the pre-trained weights and the dataset.

Shown below are the results of the model on the test_split data set.  
![image](https://user-images.githubusercontent.com/92358150/166443722-4187fb65-36b7-4425-902d-653571a62e22.png)

Some sample images with bounding boxes from the test_split and from own images are available on /test_eval and /sample_images_eval

![sample_test2](https://user-images.githubusercontent.com/92358150/166464506-87573c37-86be-4203-96f5-314f30c9ada0.png)
![007](https://user-images.githubusercontent.com/92358150/166464530-188a8764-3c74-4164-a087-8cc139a1732e.png)
