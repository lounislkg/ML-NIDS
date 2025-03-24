import pandas as pd
import numpy as np  
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data_set = pd.read_csv('archive/dataSampled.csv')
 
# Split features and labels
X = data_set.drop(' Label', axis=1)
y = data_set[' Label']  # Keep y as a 1D array for LabelEncoder and train_test_split

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values
X = X.fillna(0)

# Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply PCA to reduce dimensions (keep 99% variance)
pca = PCA(n_components=0.99)
X = pca.fit_transform(X)
print(f"Original shape: {data_set.shape}, Reduced shape: {X.shape}")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display dataset information
print("_____TRAIN_____")
print("x_train shape:", x_train.shape)
print(pd.Series(y_train).value_counts())
print("_____TEST_____")
print("x_test shape:", x_test.shape)
print(pd.Series(y_test).value_counts())

# Train DBSCAN
# eps and min_samples should be tuned for best performance
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Mark anomalies (-1 = anomaly, 1 = normal)
anomalies = np.where(dbscan_labels == -1, -1, 1)
data_set['Anomaly'] = anomalies

# Count detected anomalies
print("Anomaly count:")
print(data_set['Anomaly'].value_counts())

# Display some detected anomalies
print("Sample anomalies:")
print(data_set[data_set['Anomaly'] == -1].head())

# Convert anomalies to labels
# Map DBSCAN clusters to actual classes
malware_label = 1  # Change if needed based on dataset
anomaly_mapping = {}
for cluster in np.unique(dbscan_labels):
    mask = (dbscan_labels == cluster)
    if np.any(mask):
        majority_class = np.bincount(y[mask]).argmax()
        anomaly_mapping[cluster] = majority_class

y_pred = np.array([anomaly_mapping.get(c, malware_label) for c in dbscan_labels])

# Convert labels to binary for evaluation
y_test_binary = np.where(y_test == malware_label, 1, 0)
y_pred_binary = np.where(y_pred == malware_label, 1, 0)

# Evaluate performance
print("Confusion Matrix:\n", confusion_matrix(y_test_binary, y_pred_binary))
print("\nClassification Report:\n", classification_report(y_test_binary, y_pred_binary))
