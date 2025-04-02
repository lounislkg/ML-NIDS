import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X_fs = joblib.load('.\\pkl\\train_with_group_classes\\X_fs.pkl')
y = joblib.load('.\\pkl\\train_with_group_classes\\y.pkl')

X_fs = pd.DataFrame(X_fs)

#Replace infinite values with NaN
X_fs.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X_fs = X_fs.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size=0.2, random_state=0)

# Decision Tree training and prediction
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_train = dt.predict(X_train)
dt_test = dt.predict(X_test)


# Random Forest training and prediction
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train)
rf_train = rf.predict(X_train)
rf_test = rf.predict(X_test)   

# Extra Trees training and prediction
et = ExtraTreesClassifier(random_state = 0)
et.fit(X_train,y_train)
et_train = et.predict(X_train)
et_test = et.predict(X_test)

# XGBoost training and prediction
xg = xgb.XGBClassifier(tree_method="hist", random_state = 0)
xg.fit(X_train,y_train)
xg_train = xg.predict(X_train)
xg_test = xg.predict(X_test)
print(xg_train.shape)
base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
print(base_predictions_train.head())

dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)

dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)

x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)

stk = xgb.XGBClassifier().fit(x_train, y_train)
y_predict=stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()