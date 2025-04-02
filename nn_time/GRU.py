import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
import pandas as pd
import numpy as np

#preprocessing
dataset = pd.read_csv("MachineLearningCVE\\dataSampling_NN_GRU.csv")

batch_size = 32  # Taille de chaque batch
df_np = dataset.to_numpy()
print(df_np[0])
#shuffling 
np_shuffle = [df_np[i:i + batch_size] for i in range(0, df_np.shape[0], batch_size)]
np.random.shuffle(np_shuffle)
dataset = pd.DataFrame(np.concatenate(np_shuffle, axis=0), columns=dataset.columns)
print(dataset.head())
print(dataset.shape)

#split with datas and labels
X = dataset.drop(' Label', axis=1)
y = dataset[' Label'] #We keep y as a 1d array for labelEncoder and train_test_split

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)

print(pd.Series(y).value_counts())
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
 
#Adding the a 3rd dimension that contains the 4 previous samples for the GRU model
def create_sequence(X, y, time_steps=4):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)



#split with train and test
def split(X, y):
    X_train = np.array([])
    y_train = np.array([])
    X_test = np.array([])
    y_test = np.array([])
    X_val = np.array([])
    y_val = np.array([])
    X_pd = pd.DataFrame(X)
    y_pd = pd.Series(y)
    # y_count = y_pd.value_counts()
    # print(y_count)
    # print(len(np.where(y_pd == 4)[0]))
    y_labels = [np.where(y_pd == 0)[0], np.where(y_pd ==1)[0], np.where(y_pd ==2)[0], np.where(y_pd ==3)[0], np.where(y_pd == 4)[0], np.where(y_pd == 5)[0], np.where(y_pd == 6)[0]]
    for l in range(0, len(y_labels)):
        #append 70% of the data to the train set
        #on va essayer de prendre des batchs de 32 samples mais de mélanger les données
        #car les classes ne sont pas présentes au même endroit dans le dataset
        train_end_index= int(0.7 * len(y_labels[l]))
        X_labeled = X[y_labels[l][0:train_end_index]]
        if X_train.size == 0:
            X_train = X_labeled
        else:
            X_train = np.vstack([X_train, X_labeled])
        y_label = y[y_labels[l][0:train_end_index]]
        y_train = np.append(y_train, y_label)


        #append 15% of the data to the test set
        test_end_index = train_end_index+int(0.15 * len(y_labels[l]))
        X_labels = X[train_end_index:test_end_index] #y_labels[l][train_end_index:test_end_index]
        if X_test.size == 0:
            X_test = X_labels
        else:
            X_test = np.vstack([X_test, X_labels])
        y_label = y[y_labels[l][train_end_index:test_end_index]]
        y_test = np.append(y_test, y_label)


        #append 15% of the data to the validation set
        val_end_index = test_end_index + int(0.15 * len(y_labels[l]))
        X_labels = X[y_labels[l][test_end_index:val_end_index]]
        if X_val.size == 0:
            X_val = X_labels
        else:
            X_val = np.vstack([X_val, X_labels])
        y_label = y[y_labels[l][test_end_index:val_end_index]]
        y_val = np.append(y_val, y_label)
    return X_train, y_train, X_test, y_test, X_val, y_val

X_train, y_train, X_test, y_test, X_val, y_val = split(X, y)


from sklearn.model_selection import train_test_split
""" # Diviser le DataFrame en batches

batch_size = 32  # Taille de chaque batch

X_batches = [X[i:i + batch_size] for i in range(0, X.shape[0], batch_size)]
Y_batches = [y[i:i + batch_size] for i in range(0, y.shape[0], batch_size)]

# Mélanger les batches (pas les échantillons au sein des batches) tout en gardant une cohérence entre X et y
print(X_batches[0][:5][1])
print(Y_batches[0][:5])
indices = np.random.permutation(len(X_batches))
X_shuffled = np.array([X_batches[i] for i in indices])
y_shuffled = np.array([Y_batches[i] for i in indices])

print(X_batches[0][:5][1])
print(Y_batches[0][:5])

X = np.concatenate(X_batches, axis=0)
y = np.concatenate(Y_batches, axis=0) 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)
"""

# oversampling
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


smote=SMOTE(sampling_strategy={4: 3000, 3:500, 6: 3000}, random_state=42) # Create 1500 samples for the minority class

X_train, y_train = smote.fit_resample(X_train, y_train)

#dimension for the GRU model
X_train, y_train = create_sequence(X_train, y_train, time_steps=4)
X_test, y_test = create_sequence(X_test, y_test, time_steps=4)
X_val, y_val = create_sequence(X_val, y_val, time_steps=4)

#Removing the 24 last samples to have a multiple of 32
reduce = len(X_train) % 32
X_train = X_train[:-reduce]
y_train = y_train[:-reduce]

X_test = X_test[:-reduce]
y_test = y_test[:-reduce]


#One hot encoding for the labels
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)


#display informations
print("____TRAIN____")
print("X Train shape : ", X_train.shape)
print(y_train[:5]) 
print(pd.Series(y_train).value_counts())
print("____TEST____")
print("X test shape : ", X_test.shape)
print(y_test[:5])
print("____VAL____")
print("X val shape : ", X_val.shape)
print(y_val[:5])


def test(e=5, iter=0):
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, min_delta=0.001)
    # Modèle pour de la classification avec 7 sorties (7 classes) 
    if iter == 0:
        model = Sequential([
            GRU(64, input_shape=(4, 78), return_sequences=True),
            Dense(7, activation='softmax')
        ])
    """ if iter == 1:
        model = Sequential([
            GRU(64, input_shape=(4, 78), kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            GRU(16, input_shape=(4, 78), kernel_regularizer=l2(0.001)),
            Dense(7, activation='softmax')
        ])
    if iter == 2:
        model = Sequential([
            GRU(64, input_shape=(4, 78), kernel_regularizer=l2(0.0001)),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
    if iter == 3:
        model = Sequential([
            GRU(64, input_shape=(4, 78), kernel_regularizer=l2(0.01)),
            Dense(7, activation='softmax')
        ]) """

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',  tf.keras.metrics.Precision(name='precision_macro'), tf.keras.metrics.Recall(name='recall_macro')])
    model.summary()

    model.fit(X_train, y_train, epochs=e, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    # Calcul des métriques
    
    model.evaluate(X_test, y_test, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    print("F1 Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    
    print(classification_report(y_test_classes, y_pred_classes, target_names=labelencoder.classes_))

    
test(e=2, iter=0)