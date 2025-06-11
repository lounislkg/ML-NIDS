import joblib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

X = joblib.load('X_fs.pkl')
y = joblib.load('y.pkl')
X = pd.DataFrame(X)

labelencoder = [ "Benign", "DoS", "PortScan", "Infiltration", "webAttacks", "Brute Force", "Botnet" ]

unique, labelcount = np.unique(y, return_counts=True)
print("Name            | Label | Count")
print("--------------------------------")
for labelencode, label, count in zip(labelencoder, unique, labelcount):
    print(f"{labelencode:15} | {label:5} | {count}")
