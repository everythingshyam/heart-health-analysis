from subprocess import call
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def Model_Training():
    data = pd.read_csv("projectApp/new.csv")
    data.head()

    data = data.dropna()
    
    
    """One Hot Encoding"""

    le = LabelEncoder()
    
    data['age'] = le.fit_transform(data['age'])
    data['sex'] = le.fit_transform(data['sex'])
    data['cp'] = le.fit_transform(data['cp'])
    data['trestbps'] = le.fit_transform(data['trestbps'])
    data['chol'] = le.fit_transform(data['chol'])
    data['fbs'] = le.fit_transform(data['fbs'])
    data['restecg'] = le.fit_transform(data['restecg'])
    data['thalach'] = le.fit_transform(data['thalach'])
    data['exang'] = le.fit_transform(data['exang'])
    data['oldpeak'] = le.fit_transform(data['oldpeak'])
    data['slope'] = le.fit_transform(data['slope'])
    data['ca'] = le.fit_transform(data['ca'])
    data['thal'] = le.fit_transform(data['thal'])
   # data['target'] = le.fit_transform(data['target'])
    
    #data['EmpWorkLifeBalance'] = le.fit_transform(data['EmpWorkLifeBalance'])
    
    #data['PerformanceRating'] = le.fit_transform(data['PerformanceRating'])
    
  
    
  

    """Feature Selection => Manual"""
    x = data.drop(['target'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['target']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

    from sklearn.tree import DecisionTreeClassifier
    svcclassifier = DecisionTreeClassifier()
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    print("Confusion Matrix :")
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    
    from joblib import dump
    dump (svcclassifier,"Dec_heart.joblib")
    print("Model saved as Dec_heart.joblib")


Model_Training()
