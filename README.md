## Overview
This project presents an automated wood quality inspection system developed as a continuation of the proof of concept titled *Thesis MobileNet*. The system utilizes MobileNetV2 and MobileNetV3 architectures to classify wood defects from images, aiming to improve efficiency and consistency in mass production environments.

## Features
Automated classification of wood defects:
  - Cracks
  - Fuzz
  - Knotholes
  - No Defect

Graphical User Interface (GUI) for user interaction
Real-time image-based detection
Lightweight deep learning models (MobileNetV2 & MobileNetV3)

## Model Details
Pretrained MobileNetV2 and MobileNetV3
Custom Classifier head for wood classification
Input size: 224 x 224
Trained on 5308 wood defect images (raw dataset without augmentation)

## System Workflow
1. User inputs an image through the GUI  
2. Image is preprocessed (resized, normalized)  
3. Model predicts defect classification  
4. Result is displayed to the user  
