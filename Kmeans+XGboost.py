import pandas as pd
import numpy as np  
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load dataset
data_set = pd.read_csv('archive/dataSampled2.csv')

# Split features and labels
X = data_set.drop(' Label', axis=1)
y = data_set[' Label']

# Handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply PCA (keep 99% variance)
pca = PCA(n_components=0.99)
X = pca.fit_transform(X)
print(f"Original shape: {data_set.shape}, Reduced shape: {X.shape}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(x_train)

# Get cluster assignments
train_clusters = kmeans.predict(x_train)
test_clusters = kmeans.predict(x_test)

# Add cluster assignments as features
x_train_with_clusters = np.hstack((x_train, train_clusters.reshape(-1, 1)))
x_test_with_clusters = np.hstack((x_test, test_clusters.reshape(-1, 1)))

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(x_train_with_clusters, y_train)

# Predict with XGBoost
xgb_predictions = xgb.predict(x_test_with_clusters)

# Evaluate XGBoost model
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, xgb_predictions))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_predictions))

# Convert labels to binary (1 = malware, 0 = benign)
malware_label = 1
y_test_binary = np.where(y_test == malware_label, 1, 0)

# Convert K-Means predictions to binary
kmeans_predictions = np.where(test_clusters == malware_label, 1, 0)

# Evaluate K-Means model
print("K-Means Confusion Matrix:\n", confusion_matrix(y_test_binary, kmeans_predictions))
print("\nK-Means Classification Report:\n", classification_report(y_test_binary, kmeans_predictions))