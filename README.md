# CoE 197Z Assignment Submissions by Adonais Ray Maclang
## Assignment 3: Keyword Spotting using Transformers

Install the required packages. 

```
!pip install -r requirements.txt
```

To train, run train.py. 

```
!python3 train.py
```

To use the GUI with the pre-trained weights, run kws-infer.py

```
!python3 kws-infer.py
```



## Assignment 2: Object Detection
Faster R-CNN is an object detection model that uses a trained Region Proposal Network (RPN) algorithm to extract features. Faster R-CNN and RPN are the foundations of several 1st place entries in competitions. Read more https://arxiv.org/abs/1506.01497.

The Faster R-CNN model is finetuned using a custom drinks dataset containing 3 classes: water, coke, juice.

Install the required packages. 

```
!pip install -r requirements.txt
```

To train, run train.py. This also downloads the dataset. 

```
!python3 train.py
```

To see results on test dataset, run test.py. This uses the weights from train.py. If model_weights.pth is not available on working folder, it downloads the pre-trained weights also available in releases. This requires the drinks dataset on the folder. 

```
!python3 test.py
```

Alternatively, view the Google Colab Notebooks. 

Shown below are the results of the model on the test_split data set.  
![image](https://user-images.githubusercontent.com/92358150/166443722-4187fb65-36b7-4425-902d-653571a62e22.png)

Some sample images with bounding boxes from the test_split and from own images are available on /test_eval and /sample_images_eval

![sample_test2](https://user-images.githubusercontent.com/92358150/166464506-87573c37-86be-4203-96f5-314f30c9ada0.png)
![007](https://user-images.githubusercontent.com/92358150/166464530-188a8764-3c74-4164-a087-8cc139a1732e.png)

https://user-images.githubusercontent.com/92358150/166587931-e61c4d80-09c8-4edd-8779-bd7ef0b18369.mp4

