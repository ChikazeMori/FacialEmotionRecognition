# Facial Emotion Recognition

* Applied 2 classifiers (MLP and SVM) to the heart disease dataset and compared the results.
* Tuned hyperparameters by applying a grid search with 10-fold cross validation.
* The best MLP model had an accuracy of 0.836.
* The best SVM model had an accuracy of 0.852.

## Packages
**Python Version**: 3.8.3

**Packages used**:
* joblib==0.16.0
* matplotlib==3.2.2
* numpy==1.18.5
* pandas==1.0.5
* scikit-learn==0.23.1
* scipy==1.5.0
* seaborn==0.10.1
* skorch==0.9.0
* torch==1.7.1

**Requirements**: 
`<pip install -r requirements.txt>`

## Original Dataset
The original data was downloaded from [Real-world Affective Faces Database](http://www.whdeng.cn/raf/model1.html).


## Data cleaning
* Dummy encoded the categorical variables.
* Split the dataset into training and test sets (80% and 20%).

## EDA

**Boxplot of non-categorical variables**:

![box](https://github.com/ChikazeMori/Comparison-of-MultilayerPerceptron-and-SupportVectorMachine/blob/main/pics/boxplot.png)


**Correlation matrix**:

![corr](https://github.com/ChikazeMori/Comparison-of-MultilayerPerceptron-and-SupportVectorMachine/blob/main/pics/corr.png)

## Hyperparameters
**MLP**: 
* learning_rate=0.1
* momentum=0.85
* hidden_size=200

**SVM**: 
* C=0.1
* degree=3
* kernel=poly

## Best Models
Model |	Training score |	Validation score	| Test score
------------ | ------------- | ------------ | -----------
MLP |	0.868 |	0.860 |	0.836
SVM |	0.884 | 0.852 | 0.852

## Conclusion 
* SVM outperformed MLP
* MLP is more capable of computational training than SVM and has potential to outperform SVM when it is trained with bigger size of hidden layers.


## Specifications

### Jupyter Notebooks

All jupyter notebooks are available as ipynb.

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
