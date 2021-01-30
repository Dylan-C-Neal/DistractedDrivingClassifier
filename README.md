# Distracted Driving Classifier - VGG16 Transfer Learning
### CONTEXT

According to the CDC, ~8 people are killed in the USA each day in crashes reported to involve a
distracted driver. Distraction can take the form of visual (taking your eyes off the road), manual
(taking your hands off the wheel), or cognitive (taking your mind off driving).<sup>[1](https://www.cdc.gov/transportationsafety/distracted_driving/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fmotorvehiclesafety%2Fdistracted_driving%2Findex.html)</sup> Modern technology
such as cell phones and car-dash navigation systems have exacerbated this problem.
Auto insurance companies have a vested interest in being able to recognize these behaviors and
adjust insurance rates accordingly by the risk associated with them. Not only would this be ideal
for insurance companies, but the personal risk of an increased insurance rate would reduce
distracted driving behaviors as well, thus reducing the number of accidents caused by them.
In 2015 State Farm launched a Kaggle Competition<sup>[2](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview)</sup> with a goal to develop an image classifier
capable of categorizing photos of drivers into one of 10 different categories. One category
represents safe driving, while the other 9 represent different types of distracted driving (such as
texting or reaching in the back seat). Using Keras, a neural network model can be built to detect under what category drivers fall.
	
  
### DATA

The data consist of ~22k labeled training images and ~80k unlabeled test images. Photos were taken from the passenger seat at the same angle of a variety of driving subjects. The subjects in the training set were different than those in the test set. This split is to ensure that the model learns patterns associated with the activity rather than just overfitting to the specific subjects in the training data. There are 10 classes:
- c0: normal driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger
	
The training data exhibited not only a class imbalance, but also a subject imbalance. Classes were present from < 2,000 up to > 2,400 images in the training data. Subjects were imbalanced from < 400 up to > 1,200 images.

<p align="center">
  <img width="600" height="480" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Training_Class_Counts.png">
</p>


<p align="center">
  <img width="630" height="480" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Training_Subject_Counts.png">
</p>
  
To address this, a function was defined for balancing the subject per class composition through under and over-sampling. The mean subject/class occurrence was ~80, so the data were over/under-sampled so that each class consisted of 80 occurrences of each subject.					

### LOADING/PRE-PROCESSING
The Keras ImageDataGenerator class has many parameters for applying image pre-processing as well as randomized image adjustments within a specified range. The training and the validation/test ImageDataGenerators both rescaled the features so that all data are between 0 and 1 to standardize them, as well as applied Samplewise Center. Samplewise Center calculates the mean pixel channel intensity in a sample and then subtracts that mean from each pixel channel. This emphasizes the pieces of the photo that are distinct.

<p align="center">
  <img width="380" height="380" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Unprocessed_Photo.png"><br>
  <em>Unprocessed photo</em><br><br>
  <img width="380" height="380" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Samplewise_Center_Photo.png"><br>
  <em>Samplewise Center transformation</em>
</p>

Additionally, the training data generator had specified ranges of randomized rotation, width, height, channel, shear, zoom, and brightness shifts applied to each photo loaded in from the training data. The intention is to increase the power of the training data without collecting additional data. This helps the model generalize better and prevent overfitting.

<p align="center">
  <img width="380" height="380" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Rotation_Photo.png"><br>
  <em>Possible rotation shift with rotation_range=40</em>
</p>

### MODELING
Transfer learning was employed to build this image classifier. A VGG16 network which was trained on ImageNet was loaded via the Keras API without the final dense layers and output layer. A new dense layer and output layer was added for 10-class output with a softmax activation function. The only trainable parameters in the model were from the added layers. The loss-function utilized was categorical cross-entropy.

<p align="center">
  <img width="600" height="480" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/VGG16_Architecture.png"><br>
  <em>Unmodified VGG16 Architecture<sup><a href="https://www.researchgate.net/figure/Architecture-of-VGG16_fig1_327060416">3</a></sup></em>
</p>

### CROSS-VALIDATION ENSEMBLING
In order to keep models from overfitting to the training subjects, 4 subjects were isolated from the training data as a validation set. Each validation set includes 2 women and 2 men, all with varying shades of skin color. It is important to actively mitigate unintended racial bias in machine learning models, and this was a simple precaution to help do that. During training the validation data were frequently evaluated to make sure the model was not overfitting to the training data. The network would train for a number of steps, and then evaluate the validation data. The training loss and accuracy was compared to the validation loss and accuracy every epoch, and the weights associated with the lowest validation lost were saved to the model. 5-fold cross-validation was employed, in which 5 untrained models with the same architecture were trained on each training/validation split. Validation metrics for each CV fold are shown below.

<p align="center">
  <img width="350" height="250" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Validation_Metrics.png"><br>
  <em>Cross-Validation Metrics</em>
</p>

To generate an ensembled model, the prediction outputs from each cross-validation fold were averaged together. 

### METRICS
Because this data came from a Kaggle competition, the test data were not labeled. This was problematic for collecting more comprehensive model performance evaluation. As a small remedy, 200 test images were hand-labeled, with 20 images from each class present. The trained ensembled model was evaluated on this labeled test subset, which achieved a cross entropy loss of 0.5768. A classification report is shown below.
 
<p align="center">
  <img width="450" height="300" src="https://github.com/Dylan-C-Neal/DistractedDrivingClassifier/blob/master/report_images/Classification_Report.png"><br>
  <em>Classification Report</em>
</p>

The model had the most difficulty predicting c0 (safe driving) and c9 (talking to passenger) correctly. 80% of images were correctly classified.

Predictions for the full test set were generated using the ensembled model and submitted to Kaggle for cross entropy loss scoring. The final scores are as follows:
-	Private Score = 0.74193 – Scoreboard position of 425 out of 1,439 (70th percentile)
-	Public Score = 0.76031 – Scoreboard position of 437 out of 1,439 (69th percentile).

### FINAL THOUGHTS
Model performance was decent but could certainly be improved with more experimentation. With more dedicated GPU memory (6.0 GB was used in this project), more complex neural networks could be trained. It could be worth evaluating additional ensembling, such as feeding model predictions into another supervised learning algorithm (e.g. KNN).











