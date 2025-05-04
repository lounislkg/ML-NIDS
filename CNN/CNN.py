from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd


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

# oversampling
# Étape 1 : identifier les classes à oversampler
classes_to_oversample = [1, 4, 6]

# Étape 2 : filtrer les instances à garder telles quelles
mask_keep = ~np.isin(y_train, classes_to_oversample)
X_major = X_train[mask_keep]
y_major = y_train[mask_keep]

# Étape 3 : filtrer les instances à oversampler
mask_smote = np.isin(y_train, classes_to_oversample)
X_minor = X_train[mask_smote]
y_minor = y_train[mask_smote]

# Étape 4 : SMOTE uniquement sur les classes ciblées
# On reshape car SMOTE attend des données 2D
X_minor_2d = X_minor.reshape((X_minor.shape[0], -1))  # (n_samples, features)
smote = SMOTE(k_neighbors=3, random_state=42) 
X_minor_res, y_minor_res = smote.fit_resample(X_minor_2d, y_minor)

# Étape 5 : fusionner avec les données majoritaires
X_minor_res = X_minor_res.reshape((-1, X_train.shape[1], 1))
X_balanced = np.concatenate([X_major, X_minor_res], axis=0)
y_balanced = np.concatenate([y_major, y_minor_res], axis=0)

# Étape 6 : shuffle
from sklearn.utils import shuffle
X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

print("Nouvelles dimensions :", X_balanced.shape, y_balanced.shape)

# y_train = np.where(y_train == 0, 0, 1)  # 0 = normal, 1 = tout le reste
input_shape=(X_balanced.shape[1], 1)

# modèle Keras
model = Sequential([
    Input(shape=input_shape),
    Conv1D(16, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[ 'accuracy' ])

from sklearn.utils import compute_class_weight 

# Assure-toi que y_train contient bien des labels entiers (0 à 6)
y_balanced = y_balanced.astype(int)

class_weights = [1, 5, 1, 1, 1, 1, 6]  

print("Class weights:", class_weights)
# Convertir en dictionnaire
class_weights = dict(enumerate(class_weights))

print("X_balanced shape:", X_balanced.shape)
print("y_balanced shape:", y_balanced.shape)
model.fit(X_balanced, y_balanced, epochs=3, batch_size=32, validation_split=0.2) # class_weight=class_weights)

# évaluation scikit-learn
y_pred = model.predict(X_test)

# On prend la classe avec la probabilité la plus haute
y_pred_classes = np.argmax(y_pred, axis=1)

# Save the model
model.save('cnn_model.h5')

# Évaluation
print(classification_report(y_test, y_pred_classes))
print("____________________METRIQUES____________________")
print('Accuracy of CNN: '+ str(model.evaluate(X_test, y_test)[1]))
precision,recall,fscore,none = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted') 
print('Precision of CNN: '+(str(precision)))
print('Recall of CNN: '+(str(recall)))
print('F1-score of CNN: '+(str(fscore)))


# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
class_names = [str(i) for i in range(7)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

