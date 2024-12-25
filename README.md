# Monkey-pox-dataset-classification
This project involves creating a machine learning classifier for the Monkey pox dataset to determine whether an image is normal, Chickenpox, Measles or Monkeypox .

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#Requirements)
- [Feature Extraction](#Feature-Extraction)
- [Model Training](#Model-Training)
- [Results](#results)


## Introduction

The objective of this project is to create a robust machine learning model capable of classifying data into four distinct categories. The process includes:

Data Preprocessing: Cleaning and preparing the dataset.
Feature Extraction: Extracting meaningful features for model training.
Model Training: Training and fine-tuning the model for optimal performance.
Evaluation: Assessing the model's accuracy and effectiveness.


## Dataset 

A new skin image-based dataset for the detection of monkeypox disease. This dataset consists of four classes: Monkeypox, Chickenpox, Measles, and Normal. All the image classes are collected from internet-based sources. The entire dataset has been developed by the Department of Computer Science and Engineering, Islamic University, Kushtia-7003, Bangladesh.
It is available at (https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset).


## Requirements

Ensure the following libraries are installed:

-Python 3.x
-Jupyter Notebook
-Pandas
-NumPy
-Scikit-learn
-Matplotlib
-Seaborn
-TensorFlow or PyTorch (if applicable)
-XGBoost

 You can install the necessary packages using pip:
```bash
   pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow keras seaborn 
 ```

## Data Preprocessing

Apply image preprocessing technique to enhance the images (i.e Adaptive Equalization)
Apply data augmentation
Convert to numpy array
Normalization (i.e Min Max normalization)
Encoding

##  Feature Extraction

- Identify key features relevant to the classification task.
- Apply methods such as GLCM, Color Histogram, LBP (Local Binary Patterns), and Color Moments for extracting meaningful features.
- Combine all extracted features into a single feature set.
- Apply LASSO (Least Absolute Shrinkage and Selection Operator) for feature selection.

```python

from skimage.feature import greycomatrix, greycoprops
import numpy as np

# GLCM 
glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
contrast = greycoprops(glcm, 'contrast')

# Color Histogram 
hist = np.histogram(image.ravel(), bins=256, range=(0, 255))

# LBP 
from skimage.feature import local_binary_pattern
lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

# Color Moments
mean = np.mean(image, axis=(0, 1))
stddev = np.std(image, axis=(0, 1))
skewness = np.mean((image - mean) ** 3, axis=(0, 1))

# Combine features
combined_features = np.concatenate([contrast, hist[0], lbp.flatten(), mean, stddev, skewness])

# LASSO Feature Selection

lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train_scaled, y_train)
coefficients = lasso.coef_
selected_features = np.where(coefficients != 0)[0]

```

##  Model Training

-Split the data into training and testing sets.
-Train classification models, such as Random Forest and XGBoost, to compare performance.

Random Forest Classifier
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    selected_features, data["label"], test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
XGBoost Classifier
```python
import xgboost as xgb
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
```


## Results
The Random Forest Classifier model achieves an accuracy of 93 % on the test set for selected features. and The XGBoost Classifier achieves an accuracy of  94 %
Detailed performance metrics and plots are generated during evaluation.

