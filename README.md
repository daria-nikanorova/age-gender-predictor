# Telegram bot for age and gender prediction 

This repo contains all code of ```@AgeGenderPredictbot``` - Telegram bot that predicts age and gender by photo.

### Age and gender detection process

Age and gender detection is a three-staged process:

0. Image preprocessing 
1. Face detection
2. Age and gender prediction

For the first step we used ```cv2.dnn.blobFromImage``` method to preprocess image. This method:

* Resizes and crops image from center
* Subtracts mean values
* Scales values
* Swap Blue and Red channels

Face detection stage is based on OpenCV’s deep learning face detector.

Finally, age and gender prediction is based on the deep-convolutional neural network (CNN) that was implemented by [Levi and Hassner](https://talhassner.github.io/home/publication/2015_CVPR) in 2015. The model was trained on the [Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender).

### Technical details

This Telegram bot is:

* made with ```aiogram```
* deployed to Heroku cloud platform
* utilizes web-hook connection
