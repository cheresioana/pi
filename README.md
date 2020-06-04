# Cartoons generation
Subject: Image processing

Prof: Mircea Paul Muresan 


## Description

The aim of this project is to generate short animation clips with the help of a neural network. 
In this example the algorithm generates a short clip inspired by the well-knowed cartoons "Tom and Jerry"

The problems along the way, which also define the steps and the timeframe of the project are the following:

1. Creating a databes for tom and jerry
2. Extracting only the character from the frame, without any background
3. Teaching a model the correct sequence of characters and their interraction
4. Predicting frames
5. Composing a short video from the predicted frames

## Creating a database - Yolov2

https://github.com/thtrieu/darkflow

YOLOV2 is a state-of-the-art, real-time object detection system. Their method uses a
hierarchical view of object classification that allows them to
combine distinct datasets together.
It apply a single neural network to the full image. 
This network divides the image into regions and predicts bounding boxes and probabilities for each region. 
These bounding boxes are weighted by the predicted probabilities.

The problem encountered was that yolov2 wasn't trained to detect tom or jerry, instead it had a chaotic output.
[![Demo NotOK alpha](ressources/not_good.gif)](ressources/not_good.gif)

I created a small dataset for Tom and jerry using a project on github called yoloAnnotationTool: https://github.com/2vin/yolo_annotation_tool
In the darkflow folder there is the dataset of 336 images, along with the xml that contain the label, amd the coordinates for the surrounding rectangle.

I trained Yolov2 on the custom dataset, and after a few(a lot) of tries it worked well.
[![Demo NotOK alpha](ressources/good_classification.gif)](ressources/good_classification.gif)
