# Deep Learning Computer Vision 

## Principle Working of Single-shot detector (SSD)
### Image Classification and Object Location


Image classification in computer vision **takes an image and predicts the object in an image**, while object detection **not only predicts the object but also finds their location in terms of bounding boxes**. For example, when we build a swimming pool classifier, we take an input image and predict whether it contains a pool, while an object detection model would also tell us the location of the pool.

![image](https://user-images.githubusercontent.com/77944932/166629065-c10f50af-5cc2-4b4c-a0a3-19563766e7be.png)

### Why sliding window approach wouldn't work?

There are 2 problems:
1. **Size of window** : DIfferent types of object ans even same type of objects can be of varying sizes.

2. **Aspect ratio** : The ratio of height to width of a bounding box. A lot of objects can be present in various shapes and result in different aspect ratio.

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

Instead of using sliding window, SSD divides the image using a **grid** and have each grid cell be responsible for detecting objects in that region of the image. Detection objects simply means predicting the class and location of an object within that region. If no object is present, consider it as the background class and the location is ignored. For instance, we could use a 4x4 grid in the example below. Each grid cell is able to output the position and shape of the object it contains.

![image](https://user-images.githubusercontent.com/77944932/166630508-5a5b3be8-6629-413a-9866-cd52dc4f10cd.png)

You might be wondering what if there are multiple objects in one grid cell or we need to detect multiple objects of different shapes. There is where anchor box and receptive field come into play.

### Anchor box

Each grid cell in SSD can be assigned with multiple anchor/prior boxes. These anchor boxes are pre-defined and each one is responsible for a size and shape within a grid cell.

![image](https://user-images.githubusercontent.com/77944932/166630611-f599c60d-414a-4b8d-b5cc-1c959e4ae355.png)

SSD uses a matching phase while training, to match the appropriate anchor box with the bounding boxes of each ground truth object within an image. Essentially, the anchor box with the highest degree of overlap with an object is responsible for predicting that object’s class and its location. This property is used for training the network and for predicting the detected objects and their locations once the network has been trained. In practice, each anchor box is specified by an aspect ratio and a zoom level.

### Aspect ratio

Not all objects are square in shape. Some are longer and some are wider, by varying degrees. The SSD architecture allows pre-defined aspect ratios of the anchor boxes to account for this. The ratios parameter can be used to specify the different aspect ratios of the anchor boxes associates with each grid cell at each zoom/scale level.

![image](https://user-images.githubusercontent.com/77944932/166630811-391d00d5-6e75-47be-92da-2df7848e781c.png)

### Receptive Field

These kind of green and orange 2D array are also called feature maps which refer to a set of features created by applying the same feature extractor at different locations of the input map in a sliding window fastion. Features in the same feature map have the same receptive field and look for the same pattern but at different locations. This creates the spatial invariance of ConvNet.

![image](https://user-images.githubusercontent.com/77944932/166630935-ec8bd7ff-ef4b-4dd6-952e-06753d43f4e4.png)

As earlier layers bearing smaller receptive field can represent smaller sized objects, predictions from earlier layers help in dealing with smaller sized objects.

Because of this, SSD allows us to define a **hierarchy of grid cells** at different layers. For example, we could use a 4x4 grid to find smaller objects, a 2x2 grid to find mid sized objects and a 1x1 grid to find objects that cover the entire image.







