# Deep Learning Computer Vision 

## Principle Working of Single-shot detector (SSD)
### Image Classification and Object Location


Image classification in computer vision **takes an image and predicts the object in an image**, while object detection **not only predicts the object but also finds their location in terms of bounding boxes**. For example, when we build a swimming pool classifier, we take an input image and predict whether it contains a pool, while an object detection model would also tell us the location of the pool.

![image](https://user-images.githubusercontent.com/77944932/166629065-c10f50af-5cc2-4b4c-a0a3-19563766e7be.png)

### Why sliding window approach wouldn't work?

There are 2 problems:
1. **Size of window** : DIfferent types of object ans even same type of objects can be of varying sizes.

2. **Aspect ratio** : The ratio of height ro width of a bounding box. A lot of objects can be present in various shapes and result in different aspect ratio.

![slidingwindow](https://user-images.githubusercontent.com/77944932/166629575-6ec5301a-816c-4436-87a6-765e6b57ca11.gif)
Example of sliding window approach

There are 2 types of mainstream object detection algoritm.

1) R-CNN and Fast(er) R-CNN first to identify regions where objects are expected to be found and then detect objects only in those regions using convnet. This region proposal algorithms usually have slightly better accuracy but slower to run.

2) YOLO (You Only Look Once) and SSD (Single-Shot Detector) use a fully convolutional approach in which the network is able to find all objects within an image in one pass (hence ‘single-shot’ or ‘look once’) through the convnet. Single-shot algorithms are more efficient and has as good accuracy.

### Single-Shot Detector (SSD)

SSD has two components: **a backbone model** and **SSD head** Backbone model usually is a pre-trained image classification network as a feature extractor. This is typically a network like ResNet trained on ImageNet from which the final fully connected classification layer has been removed. We are thus left with a deep neural network that is able to extract semantic meaning from the input image while preserving the spatial structure of the image albeit at a lower resolution. The SSD head is just one or more convolutional layers **added to this backbone and the outputs are interpreted as the bounding boxes and classes of objects in the spatial location of the final layers activations.**

In the figure below, the first few layers (white boxes) are the backbone, the last few layers (blue boxes) represent the SSD head.

![image](https://user-images.githubusercontent.com/77944932/166630254-df602e34-5d26-4f45-b213-98a2114c50ea.png)

### Grid Cell



