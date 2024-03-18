from subprocess import call

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv("new.csv")
data.head()

data = data.dropna()

data.isnull().sum()

# Statistical summary
data.describe().T

x = data.drop(['target'], axis=1)
data = data.dropna()

print(type(x))
y = data['target']
print(type(y))
x.shape

# Outcome countplot
import seaborn as sns
sns.countplot(x = 'target',data = data)

data.hist()

from sklearn.feature_selection import mutual_info_classif
importance = mutual_info_classif(x,y)
feat_importance = pd.Series(importance, data.columns[0: len(data.columns)-1])
feat_importance.plot(kind='barh', color='teal')