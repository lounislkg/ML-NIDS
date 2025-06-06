from os import pipe
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# ----- Prétraitement -----
# dataset = pd.read_csv('MachineLearningCVE/dataSampled_Grouped.csv')

#split with datas and labels
# X = dataset.drop(' Label', axis=1)
# y = dataset[' Label'] #We keep y as a 1d array for labelEncoder and train_test_split

X_fs = joblib.load('X_fs.pkl')
y = joblib.load('y.pkl')
X = pd.DataFrame(X_fs)

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# ----- Modèles de base -----

base_learners = [
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("et", ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ("xgb1", XGBClassifier(random_state=42))
]

# ----- Meta-modèle (empilé) -----

meta_model = XGBClassifier(random_state=42)

# ----- Stacking -----

stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,  # Cross-validation interne pour les base learners
    n_jobs=-1,  # Utilise tous les cœurs CPU
    passthrough=False  # Si True, ajoute X original aux features du meta-model
)

# ----- Pipeline -----

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('stack', stack_model)
])

# ----- Entraînement et évaluation -----

pipeline.fit(X_train, y_train)

start = time.time()
y_pred = pipeline.predict(X_test)
end = time.time()

# Pour mesurer le temps d'inférence précisement
# import timeit
# # timeit exécute plusieurs fois et donne une mesure plus stable
# exec_time = timeit.timeit(lambda: pipeline.predict(new_data), number=10)
# print(f"Temps moyen pour 10 inférences : {exec_time / 10:.6f} secondes")

# Enregistrer le modèle
joblib.dump((pipeline, labelencoder), 'stacking_model.joblib')  # Extension recommandée

print(f"Temps d'inférence pour {len(X_test)} échantillons : {end - start:.6f} secondes")
print(f"Temps moyen par échantillon : {(end - start)/len(X_test):.6f} secondes")

# ----- Affichage
print("Rapport de classification du modèle empilé")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
class_names = [str(i) for i in range(7)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

# ------ Test du modèle ------
X_batch = joblib.load('X_fs.pkl')
X = pd.DataFrame(X_batch)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)


batch = X_batch[0].reshape(1, -1)
print(batch)
prediction = pipeline.predict(batch)
print(prediction)
print(labelencoder.inverse_transform(prediction))

