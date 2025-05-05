import pandas as pd
import numpy as np
import joblib
import os

dataset = pd.read_csv('MachineLearningCVE/dataSampled_Grouped.csv')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# oversampling
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy={4:500}, random_state=42) # Create 1500 samples for the minority class

X_train, y_train = smote.fit_resample(X_train, y_train)

#display informations
print("____TRAIN____")
print("X Train shape : ", X_train.shape)
print(pd.Series(y_train).value_counts())
print("____TEST____")
print("X test shape : ", X_test.shape)
print(pd.Series(y_test).value_counts())
#joblib.dump(y, 'y.pkl')
#train models
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV


if(os.path.exists('dt.pkl')):
    dt=joblib.load('dt.pkl')
else:
    dt=DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
# dt_score=dt.score(X_test,y_test)
# y_predict=dt.predict(X_test)
# y_true=y_test
# print('Accuracy of DT: '+ str(dt_score))
# precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
# print('Precision of DT: '+(str(precision)))
# print('Recall of DT: '+(str(recall)))
# print('F1-score of DT: '+(str(fscore)))
# print(classification_report(y_true,y_predict))
# cm=confusion_matrix(y_true,y_predict)
# f,ax=plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()
# dt_train = dt.predict(X_train)
# dt_test = dt.predict(X_test)

# joblib.dump(dt, 'dt.pkl') #pour sauvegarder le modèle



# Random Forest training and prediction
if (os.path.exists('rf.pkl')):
    rf=joblib.load('rf.pkl')
else:
    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) 
# rf_score=rf.score(X_test,y_test)
# y_predict=rf.predict(X_test)
# y_true=y_test
# print('Accuracy of RF: '+ str(rf_score))
# precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
# print('Precision of RF: '+(str(precision)))
# print('Recall of RF: '+(str(recall)))
# print('F1-score of RF: '+(str(fscore)))
# print(classification_report(y_true,y_predict))
# cm=confusion_matrix(y_true,y_predict)
# f,ax=plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()  
# rf_train = rf.predict(X_train)
# rf_test = rf.predict(X_test)
# joblib.dump(rf, 'rf.pkl') #pour sauvegarder le modèle


# Extra trees training and prediction
if (os.path.exists('et.pkl')):
    et=joblib.load('et.pkl')
else:
    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
# et_score=et.score(X_test,y_test)
# y_predict=et.predict(X_test)
# y_true=y_test
# print('Accuracy of ET: '+ str(et_score))
# precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
# print('Precision of ET: '+(str(precision)))
# print('Recall of ET: '+(str(recall)))
# print('F1-score of ET: '+(str(fscore)))
# print(classification_report(y_true,y_predict))
# cm=confusion_matrix(y_true,y_predict)
# f,ax=plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()
#
# et_train = et.predict(X_train)
# et_test = et.predict(X_test)
# joblib.dump(et, 'et.pkl') #pour sauvegarder le modèle


""" 
# Calcul des poids pour chaque classe
print(y_train)

class_weights = [int(len(y_train) / len(y_train[y_train == y_train[c]])) for c in range(0,len(y_train))]
print(class_weights)

param_grid = {
    'max_depth': [3, 6, 9, None],
    'learning_rate': [0.01, 0.1, 0.3, None],
    'n_estimators': [10, 20, 50],
}

grid = GridSearchCV(XGBClassifier(), param_grid, scoring='f1_weighted', cv=3, n_jobs=-1, error_score='raise')
grid.fit(X_train, y_train, sample_weight=class_weights)
print(grid.best_params_)
"""




# XGboost training and prediction
if (os.path.exists('xg.pkl')):
    xg=joblib.load('xg.pkl')
else:   
    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
# xg_score=xg.score(X_test,y_test)
# y_predict=xg.predict(X_test)
# y_true=y_test
# default_params = xg.get_params()
# print("Default parameters: ", default_params)
# print('Accuracy of XGBoost: '+ str(xg_score))
# precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
# print('Precision of XGBoost: '+(str(precision)))
# print('Recall of XGBoost: '+(str(recall)))
# print('F1-score of XGBoost: '+(str(fscore)))
# print(classification_report(y_true,y_predict))
# cm=confusion_matrix(y_true,y_predict)
# f,ax=plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()
# xg_train = xg.predict(X_train)
# xg_test = xg.predict(X_test)
# joblib.dump(xg, 'xg.pkl') #pour sauvegarder le modèle


#Feature Selection 
# Save the feature importance lists generated by four tree-based algorithms
dt_feature = dt.feature_importances_
rf_feature = rf.feature_importances_
et_feature = et.feature_importances_
xgb_feature = xg.feature_importances_

avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature) / 4
feature=(dataset.drop([' Label'],axis=1)).columns.values
f_list = tuple(zip(map(lambda x: round(x, 4), avg_feature), feature)) # sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)
selected_df = pd.DataFrame(f_list, columns=["Feature", "Importance"])
selected_df.to_csv("ordered_features.csv", index=False)
exit(0)
print('Feature list : ' ,f_list)
# Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
Sum = 0
fs = []
selected_features = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
    selected_features.append((f_list[i][1], f_list[i][0]))
    if Sum>=0.9:
        break
X_fs = dataset[fs].values

# Sauvegarde dans un fichier CSV
selected_df = pd.DataFrame(selected_features, columns=["Feature", "Importance"])
selected_df.to_csv("ordered_features.csv", index=False)

# X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
# print(X_train.shape)
# joblib.dump(X_fs, 'X_fs.pkl')
