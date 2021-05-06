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
|CNN|8557.9|**0.74**

## Conclusion 
* SVM outperformed MLP
* MLP is more capable of computational training than SVM and has potential to outperform SVM when it is trained with bigger size of hidden layers.


## Specifications

### Jupyter Notebooks

All jupyter notebooks are available as ipynb and designed to be run on Google Colab.

* data_preparation: Clean the dataset
* MLP_optimisation: Apply a grid search for MLP
* SVM_optimisation: Apply a grid search for SVM
* MLP_testing: Test the MLP model
* SVM_testing: Test the SVM model

### Folders

* Code: Contains all of the codes implemented

### Files

* MLP_optimised.joblib: The optimised MLP model
* SVM_optimised.joblib: The optimised SVM model
* Report: The report for the whole project
