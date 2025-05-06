import joblib
import json
import numpy as np
import pandas as pd

import order_features as of



# ----- Chargement du modèle -----

pipeline, le = joblib.load('stacking_model.joblib')


# ----- Lancement de l'API -----
from fastapi import FastAPI, Request
from pydantic import BaseModel

class Data(BaseModel):
    data: list

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

#@app.post("/predict")
async def predict(df: pd.DataFrame):
    # Convertir les données d'entrée en DataFrame
    # print(df.shape)
    # print(df.head(2))
    # Remplacer les valeurs infinies par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remplir les valeurs NaN
    df = df.fillna(0)
    # print(pipeline)
    # Normaliser les données
    # df = pipeline['scaler'].transform(df)
    np_arr = df.to_numpy()
    # Prédire avec le modèle 
    print(np_arr)
    prediction = pipeline.predict(np_arr)
    predict_proba = pipeline.predict_proba(np_arr)
    # print(prediction)

    return (prediction.tolist(), predict_proba.tolist())

@app.post('/flow')
async def flow(request: Request):
    body = await request.body()
    data_dict = json.loads(body.decode('utf-8'))
    df = of.reorder_features(data_dict)
    # prediction = await predict(df)
    # print(prediction)
    return {"message": "ok"}


@app.post("/test")
async def test(request: Request):
    body = await request.body()
    data_dict = json.loads(body.decode('utf-8'))
    df = pd.DataFrame([data_dict])
    prediction = await predict(df)
    print(prediction)
    return {"message": "ok"}

# sudo .venv/bin/cicflowmeter -i wlo1 -u http://localhost:8000/flow -v \
# --fields pkt_len_var,fwd_pkt_len_max,subflow_fwd_byts,psh_flag_cnt,bwd_pkt_len_std,totlen_fwd_pkts,fwd_act_data_pkts,dst_port,bwd_pkts_s,fwd_iat_max,bwd_pkt_len_mean,bwd_seg_size_avg,pkt_len_std,pkt_size_avg,pkt_len_mean,ack_flag_cnt,pkt_len_max,bwd_pkt_len_max,fwd_seg_size_avg,init_fwd_win_byts,bwd_pkt_len_min,fwd_pkt_len_mean,flow_byts_s,tot_bwd_pkts,tot_fwd_pkts,flow_duration,subflow_fwd_pkts,flow_iat_max,fwd_iat_std,subflow_bwd_byts,fwd_iat_tot,flow_iat_std



    
