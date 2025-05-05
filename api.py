import joblib
import numpy as np
import pandas as pd

# ----- Chargement du modèle -----

pipeline, le = joblib.load('stacking_model.joblib')



# ----- Lancement de l'API -----
from fastapi import FastAPI
from pydantic import BaseModel

class Data(BaseModel):
    data: list

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(data: Data):
    # Convertir les données d'entrée en DataFrame
    df = pd.DataFrame(data.data)
    print(df.shape)
    print(df.head(2))
    # Remplacer les valeurs infinies par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remplir les valeurs NaN
    df = df.fillna(0)
    print(pipeline)
    # Normaliser les données
    # df = pipeline['scaler'].transform(df)

    # Prédire avec le modèle 
    prediction = pipeline.predict(df)
    print(prediction)

    return {"prediction": prediction.tolist()}
