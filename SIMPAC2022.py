import pandas as pd
import numpy as np

dataset = pd.read_csv('MachineLearningCVE\\dataSampled.csv')

#split with datas and labels
X = dataset.drop(' Label', axis=1)
y = dataset[' Label'] #We keep y as a 1d array for labelEncoder and train_test_split

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)


#scaling datas
#Choisir le bon scaler, simpca2022 utilise MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)


#transform labels to numbers
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()  # Création de l'encodeur
y = labelencoder.fit_transform(y)  # Encodage de la dernière colonne


#split with train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#display informations
print("____TRAIN____")
print("X Train shape : ", X_train.shape)
print(pd.Series(y_train).value_counts())
print("____TEST____")
print("X test shape : ", X_test.shape)
print(pd.Series(y_test).value_counts())

#oversampling
from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"
X_train, y_train = smote.fit_resample(X_train, y_train)



