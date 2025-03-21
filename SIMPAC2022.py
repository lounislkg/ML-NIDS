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

#oversampling
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy={8:500, 9:1500, 13:1000}, random_state=42) # Create 1500 samples for the minority class 8, 9, 13

X_train, y_train = smote.fit_resample(X_train, y_train)


#display informations
print("____TRAIN____")
print("X Train shape : ", X_train.shape)
print(pd.Series(y_train).value_counts())
print("____TEST____")
print("X test shape : ", X_test.shape)
print(pd.Series(y_test).value_counts())

#train models
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
import seaborn as sns
""" 
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show() """

# Random Forest training and prediction
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train) 
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()