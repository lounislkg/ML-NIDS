# Chargement
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

dataset = pd.read_csv('../MachineLearningCVE/dataSampled_Grouped.csv')
#split with datas and labels
X = dataset.drop(' Label', axis=1)
y = dataset[' Label'] #We keep y as a 1d array for labelEncoder and train_test_split

#transform labels to numbers
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()  # Création de l'encodeur
y = labelencoder.fit_transform(y)  # Encodage de la dernière colonne

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)


# Suppose X a la forme (n_samples, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# reshape pour CNN 1D : (samples, time steps, channels)
X_cnn = X_scaled[..., np.newaxis]  # shape (n_samples, n_features, 1)

# split
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42, stratify=y)

model = load_model('cnn_model.h5')

# Pour mesurer le temps d'inférence précisement
import timeit

exec_time = timeit.timeit(lambda: model.predict(X_test), number=10)
print(f"Temps moyen pour 10 inférences : {exec_time / 10:.6f} secondes / inférence ")
print("Temps total pour 10 inférences : ", exec_time)

# Afficher la matrice de confusion
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Classification report : \n", classification_report(y_test, y_pred_classes))
cm = confusion_matrix(y_test, y_pred_classes)
class_names = [str(i) for i in range(7)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()
