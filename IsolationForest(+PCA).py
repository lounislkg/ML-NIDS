import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

scaler = StandardScaler()


df_original = pd.read_csv('webattack.csv')


#enleve toutes les colonnes qui ne sont pas de type float64 en les remplacant par des NaN
df_original.replace([np.inf, -np.inf], np.nan, inplace=True)
#enleve toutes les colonnes vides 
df_original.dropna(inplace=True)

# print(df.info())
# print(df.head()) 

#enlève la colonne label pour ne pas entrainer le modele avec les labels donnés 
df = df_original.drop(columns=[' Label'])

#df.drop(df.index[0], inplace=True)
#print(df.dtypes)

print(np.isfinite(df).all().all())  # Doit retourner True



df_scaled = scaler.fit_transform(df)  # Centre les données (moyenne 0, variance 1)

pca = PCA(n_components=32, random_state=42) 
df_pca = pca.fit_transform(df_scaled)
#determine la dimensions minimale pour avoir 99% de la variance (Il faut set n_components à 78)
# print(np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.99) + 1) # Doit retourner 32
print("PCA Done")
isoforest = IsolationForest(n_estimators=1000,contamination=0.01, random_state=12)  # contamination = proportion d'anomalies attendue
isoforest.fit(df_scaled)   

# Prédire les anomalies (1 = normal, -1 = anomalie)
anomalies = isoforest.predict(df_scaled)

print("Isolation Forest Done")


# from sklearn.neighbors import LocalOutlierFactor
# lof = LocalOutlierFactor(n_neighbors=100, contamination=0.05)
# anomalies_lof = lof.fit_predict(df_scaled)
# anomalies_lof_pca = lof.fit_predict(df_pca)
# print(f"Anomalies détectées par LOF avec 100 voisins: {sum(anomalies_lof == -1)}")
# print(f"Anomalies détectées par LOF avec 100 voisins après PCA: {sum(anomalies_lof_pca == -1)}")

# print("Local Outlier Factor Done")
"""Partie vérification des résultats (pas tres probateur pr le moment)"""

mot = "Web Attack"
count = df_original[' Label'].str.contains(mot, case=False, na=False).sum()

print(f"Le mot '{mot}' apparaît {count} fois.")

print("Longeur de anomalies: ", len(anomalies)) 
print("Longeur de df_original: ", len(df_original))
print("Longeur de df: ", len(df))  
print("Longeur de df_pca: ", len(df_pca))
print("Longeur de df_scaled: ", len(df_scaled))

# Compter le nombre d'anomalies
print(pd.Series(anomalies).value_counts())  # Doit retourner

correct = 0
incorrect = 0
for i in range(0, len(anomalies)):
    if anomalies[i] == -1:
        if "Web Attack" in df_original.iloc[i][' Label']:
            print(df_original.iloc[i][' Label'])
            correct += 1
        else:
            if i < 100:
                print(f"Column {i} flow duration : {df_original.iloc[i][' Flow Duration']},  Active Std: {df_original.iloc[i][' Active Std']} label: {df_original.iloc[i][' Label']}")
            incorrect += 1

print("Correct: ", correct)
print("Incorrect: ", incorrect)
print("Accuracy: ", correct / (correct + incorrect))