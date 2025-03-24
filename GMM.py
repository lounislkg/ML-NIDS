import pandas as pd
import numpy as np  
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('archive/dataSampled.csv')
 
#split with datas and labels
X = data_set.drop(' Label', axis=1)
y = data_set[' Label'] #We keep y as a 1d array for labelEncoder and train_test_split

#Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

#fill NaN values
X = X.fillna(0)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform(y)


# Apply PCA to reduce dimensions (keep 99% of explained variance)   
pca = PCA(n_components=0.99)
X = pca.fit_transform(X)  # Transform the dataset
print(f"Original shape: {data_set.shape}, Reduced shape: {X.shape}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#display information about the dataset
print("_____TRAIN_____")
print("x_train shape: ", x_train.shape)
print(pd.Series(y_train).value_counts())
print("_____TEST_____")
print("x_test shape: ", x_test.shape)  
print(pd.Series(y_test).value_counts())

# Remove the second application of PCA on x_train and x_test
# x_train = pca.fit_transform(x_train)            
# x_test = pca.transform(x_test)  

from sklearn.mixture import GaussianMixture   
#Gaussian Process   
# Train GMM with 3 components (can be adjusted)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Compute the log-likelihood score of each sample
scores = gmm.score_samples(X)

# Define an anomaly threshold (5% lowest scores)
threshold = np.percentile(scores, 5)

# Mark anomalies (-1 for anomaly, 1 for normal)
anomalies = np.where(scores < threshold, -1, 1)

# Add anomaly labels to the dataframe
data_set['Anomaly'] = anomalies

# Count detected anomalies
print("Anomaly count:")
print(data_set['Anomaly'].value_counts())

# Display some detected anomalies
print("Sample anomalies:")
print(data_set[data_set['Anomaly'] == -1].head())


from sklearn.metrics import accuracy_score

# Étape 2: Prédictions des clusters
train_clusters = gmm.predict(x_train)
test_clusters = gmm.predict(x_test)

# Étape 3: Associer clusters aux vraies classes
# Trouver la classe majoritaire pour chaque cluster
mapping = {}
for cluster in np.unique(train_clusters):
    mask = (train_clusters == cluster)
    majority_class = np.bincount(y_train[mask]).argmax()
    mapping[cluster] = majority_class

# Transformer les clusters en labels prédits
y_train_pred = np.array([mapping[c] for c in train_clusters])
y_test_pred = np.array([mapping[c] for c in test_clusters])

# Étape 4: Calculer l'accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Accuracy on training set: {train_accuracy:.2f}")
print(f"Accuracy on test set: {test_accuracy:.2f}")

from sklearn.metrics import classification_report, confusion_matrix

# Convert labels to binary (1 = malware, 0 = benign)
malware_label = 1  # Change if needed based on how malware is labeled in y_test
y_test_binary = np.where(y_test == malware_label, 1, 0)

# Convert GMM anomalies to binary (1 = malware, 0 = benign)
gmm_predictions = np.where(y_test_pred == malware_label, 1, 0)

# Compare Predictions with Ground Truth
print("Confusion Matrix:\n", confusion_matrix(y_test_binary, gmm_predictions))
print("\nClassification Report:\n", classification_report(y_test_binary, gmm_predictions))
