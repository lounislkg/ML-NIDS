import joblib
import numpy as np 
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

X_fs = joblib.load('X_fs.pkl')
y = joblib.load('y.pkl')
X = pd.DataFrame(X_fs)

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)

sc = MinMaxScaler()
X = sc.fit_transform(X)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# load model
if(os.path.exists('stacking_model.joblib')):
    stack_model, le =joblib.load('stacking_model.joblib')
else:
    print("Model not found")
    exit(1)

# Pour mesurer le temps d'inférence précisement
import timeit
# timeit exécute plusieurs fois et donne une mesure plus stable
exec_time = timeit.timeit(lambda: stack_model.predict(X_test), number=10)
print(f"Temps moyen pour 10 inférences : {exec_time / 10:.6f} secondes / inférence ")
print("Temps total pour 10 inférences : ", exec_time)

# Afficher la matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = stack_model.predict(X_test)
print("Classification report : \n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
class_names = [str(i) for i in range(7)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

