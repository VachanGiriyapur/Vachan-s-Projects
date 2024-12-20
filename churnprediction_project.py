# -*- coding: utf-8 -*-
"""ChurnPrediction_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GrZPYIJofLfWhcRUg7ejdn6O13s7Dk3f
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

"""# **Exploratary** **Data** **Analysis**"""

df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
print("******************************")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0).astype('float64')
df.info()
df.isnull().sum()

df['Churn'].value_counts()
sns.histplot(df['Churn'])

sns.histplot(df['PaymentMethod'])

sns.histplot(df['StreamingTV'])

sns.histplot(df['TechSupport'])

sns.histplot(df['DeviceProtection'])

sns.histplot(df['OnlineBackup'])

sns.histplot(df['OnlineSecurity'])

sns.histplot(df['InternetService'])

sns.histplot(df['MultipleLines'])

sns.histplot(df['PhoneService'])

sns.histplot(df['Contract'])

sns.histplot(df['StreamingMovies'])

sns.boxplot(df['tenure'])

sns.boxplot(df['MonthlyCharges'])

sns.boxplot(df['TotalCharges'])

sns.histplot(df['Churn'])

"""# **Data Preprocessing**"""

#One-Hot Encoding
categorical_features = ['PaymentMethod', 'InternetService','StreamingMovies',
                        'Contract','MultipleLines','OnlineSecurity',
                        'OnlineBackup','DeviceProtection', 'TechSupport',
                        'StreamingTV', 'PhoneService']
df = pd.get_dummies(df, columns=categorical_features, drop_first= True)

#LabelEncoding
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['Partner'] = le.fit_transform(df['Partner'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['PaperlessBilling'] = le.fit_transform(df['PaperlessBilling'])
df['Churn'] = le.fit_transform(df['Churn'])
df.info()
print(df.head())

scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

x = df[['Dependents', 'tenure', 'PhoneService_Yes',
       'MultipleLines_Yes','OnlineSecurity_Yes', 'OnlineBackup_Yes',
       'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
       'PaperlessBilling', 'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'InternetService_Fiber optic', 'InternetService_No',
       'MonthlyCharges', 'TotalCharges']]
y = df[['Churn']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

"""# **Support Vector Machine**"""

#Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot()

y_score = svm_model.decision_function(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Ture Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()

importance = svm_model.coef_[0]
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': np.abs(importance)})
feature_importance = feature_importance.sort_values(by = 'Importance', ascending = False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - SVM')
plt.show()

"""# **Logistic Regression**"""

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_score = model.decision_function(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Ture Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()

importance = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': np.abs(importance)})
feature_importance = feature_importance.sort_values(by = 'Importance', ascending = False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - LR')
plt.show()

"""# **Random Forest**"""

#RandomForest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_score = rf_model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Ture Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()

plt.barh(x_train.columns, rf_model.fit(x_train,y_train).feature_importances_)
plt.show()

"""# **Neural Network**"""

import tensorflow as tf
from tensorflow import keras

mlp_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

y_pred = mlp_model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.show()

y_score = model.decision_function(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Ture Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

weights = mlp_model.layers[0].get_weights()[0]
importance = np.abs(weights).sum(axis=1)
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': np.abs(importance)})
feature_importance = feature_importance.sort_values(by = 'Importance', ascending = False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - MLP')
plt.show()

"""## **XG Boost Model**"""

import xgboost as xgb
import seaborn as sns

xgb_model = xgb.XGBClassifier(n_estimators=2500, use_label_encoder = False, eval_metric = 'logloss', random_state = 42)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
y_pred_proba = xgb_model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def plot_confusion_matrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

plot_confusion_matrix(y_test, y_pred)

def plot_roc_curve(y_test, y_pred_proba):
  fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
  plt.figure(figsize=(10, 6))
  plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
  plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title("ROC Curve")
  plt.legend(loc = "lower right")
  plt.show()

plot_roc_curve(y_test, y_pred_proba)

plt.barh(x_train.columns, xgb_model.fit(x_train,y_train).feature_importances_)
plt.show()