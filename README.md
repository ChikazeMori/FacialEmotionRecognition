# Facial Emotion Recognition

* Implemented 3 feature descriptors (SIFT, ORB, HOG) and 3 classifiers (SVM, KNN, MLP).
* Applied every combination of these algorithms for facial emotion recognition.
* Applied Convolutional Neural Network for facial emotion recognition.
* Compared the results of all the models.
* Tested the best model on a video.

## Packages
**Python Version**: 3.8.3

**Main Packages used**:
* torch==1.8.1
* torchvision==0.9.1
* scikit-image==0.18.1
* opencv-python==4.5.1.48

**Requirements**: 
`<pip install -r requirements.txt>`

## Original Dataset
The original data was downloaded from [Real-world Affective Faces Database](http://www.whdeng.cn/raf/model1.html).

## Performances
Feature Descriptor | Classifier |	Time (second)	| Accuracy
------------ | ------------- | ------------ | --------
SIFT |SVM |	101.0 + 237.7 |0.42
SIFT |SVM |	101.0 + 9.8 | 0.36
SIFT |MLP|	101.0 + 130.6| 0.42
ORB |SVM |41.3 + 219.0 |0.39
ORB |SVM |	41.3 + 9.7 | 0.28
ORB |MLP|	41.3 + 133.6| 0.38
HOG |SVM |102.3 + 441.4|0.65
HOG |SVM |	102.3 + 12.2| 0.53
HOG |MLP|	102.3 + 208.2 | 0.62
-|CNN|8557.9|**0.74**

## Examples of Results

**Examples of facial emotion recognition by the HOG- SVM model**:

![Examples of facial emotion recognition by the HOG- SVM model](https://github.com/ChikazeMori/Facial-Emotion-Recognition/blob/main/examples/SVM_HOG.png)


**Example of facial emotion recognition by the CNN model on a video**:

![Example of facial emotion recognition by the CNN model on a video](https://github.com/ChikazeMori/Facial-Emotion-Recognition/blob/main/examples/video.gif)

## Conclusion 
* CNN significantly outperforms the other models although it requires a lot of computation.

## Specifications

### Jupyter Notebooks

All jupyter notebooks are available as ipynb and designed to be run on **Google Colab**.

* **test_functions**: Test EmotionRecognition and EmotionRecognitionVideo functions and displaying the results

### Folders

* CW_Folder_ChikazeMori_200038013: Contains all the files needed for this projeect
* /**Code**: Contains all of the codes implemented
* /Models: Stores all the models trained
* /Video: Contains all the videos produced as outputs

### Files

* Code/**EmotionRecognition.py**: Displays 4 random images from the test set with the model’s predictions and the true labels
* Code/**EmotionRecognitionVideo.py**: Displays the results of CNN model tested on a video
* Report: The report for the whole project
