# DeepFake-Detection-Model
## Introduction
This projects focuses in detection of deepfakes videos using various deep learning techniques. We have compared also compared our results by training & analyzing different DeepLearning models. We have achived deepfake detection by using transfer learning where we have choosen one of the pretrained DeepLearning Model that is used to obtain a feature vector, further the LSTM layer is trained using the features.

## System Architecture

![image](https://github.com/himansh19/DeepFake-Detection-Model/assets/89848299/a2aef78c-b142-40e9-926d-d73e53712a5d)

 ## Dataset 
  - [FaceForensics++](https://github.com/ondyari/FaceForensics)
  - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
  - [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)

## Preprocessing
  - Load the dataset
  - Split the video into frames
  - Face Detection
  - Croping the face from each frame
  - Creating the new face cropped video
  - Save the face cropped video
    
## Model and train
  - It will load the preprocessed video and labels from a csv file.
  - Create a pytorch model using transfer learning with RestNext50 and LSTM.
  - Split the data into train and test data
  - Train the model
  - Test the model
  
