import numpy as np
import pandas as pd
import joblib

X_fs = joblib.load('X_fs.pkl')
y = joblib.load('y.pkl')
X = pd.DataFrame(X_fs)

batch = X.sample(50, random_state=42)

# request localhost 
import requests

url = 'http://localhost:8000/predict'
myobj = {'data': batch.to_dict(orient='records')}
print(myobj)

x = requests.post(url, json = myobj)

print(x.text)
