import pandas as pd
import numpy as np

"""
    Test à faire : 
    Ne pas modifier l'odre temporel des données, 
    Si on mélange les données il devient compliqué d'utiliser le temps comme feature
    Problème:  
    Des opérations itératives sur un DataFrame peuvent être lentes, surtout si le DataFrame est grand.
    Donc on va utiliser une approche vectorisée pour créer le masque.
"""

file_path = [
             "Tuesday-WorkingHours.pcap_ISCX.csv",
             "Wednesday-workingHours.pcap_ISCX.csv",
             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
             "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
             "Friday-WorkingHours-Morning.pcap_ISCX.csv",
             "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
             "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            ]

df_s = pd.DataFrame()
for file in file_path:
    df = pd.read_csv("MachineLearningCVE\\" + file)
    print("Fichier : ", file)
    benign_rows = df[' Label'] == 'BENIGN'
    mask = [not (i % 3 < 2) if benign else True for i, benign in enumerate(benign_rows)]
    df = df[mask]
    df.loc[df[' Label'] == 'BENIGN', ' Label'] = 0
    # Liste des conditions à vérifier
    conditions = [
        (df[' Label'] == 'DoS slowloris', 1),
        (df[' Label'] == 'DoS Hulk', 1),
        (df[' Label'] == 'DoS GoldenEye', 1),
        (df[' Label'] == 'DDoS', 1),
        (df[' Label'] == 'DoS Slowhttptest', 1),
        (df[' Label'] == 'PortScan', 2),
        (df[' Label'] == 'Infiltration', 3),
        (df[' Label'] == 'Heartbleed', 3),
        (df[' Label'] == 'Web Attack � Brute Force', 4),
        (df[' Label'] == 'Web Attack � XSS', 4),
        (df[' Label'] == 'Web Attack � Sql Injection', 4),
        (df[' Label'] == 'SSH-Patator', 5),
        (df[' Label'] == 'FTP-Patator', 5),
        (df[' Label'] == 'Bot', 6)
    ]
    for condition, value in conditions:
        if condition.any():
            df.loc[condition, ' Label'] = value
    df_s = pd.concat([df_s, df])

print(df_s.head())
print(df_s[' Label'].value_counts())
print("taille de df_s: ", len(df_s))
df_s.to_csv("MachineLearningCVE\\dataSampling_NN_GRU.csv", index=False)