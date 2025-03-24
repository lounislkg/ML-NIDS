import pandas as pd
import numpy as np

file_path = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
             "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
             "Friday-WorkingHours-Morning.pcap_ISCX.csv",
             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
             "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
             "Tuesday-WorkingHours.pcap_ISCX.csv",
             "Wednesday-workingHours.pcap_ISCX.csv"]

labels = []
words = []
wordsCount = dict()
df_s = pd.DataFrame()
for file in file_path:
    df = pd.read_csv("MachineLearningCVE\\" + file)
    words.append(df[' Label'].unique())
    counts = df[" Label"].value_counts()
    #labels.append(counts)
    for word in df[' Label'].unique():
        if word not in wordsCount:
            wordsCount[word] = int(counts.get(word))
        else:
            wordsCount[word] += int(counts.get(word))
    #Déterminer si modifier la fréquence des échantillons à un impact sur la précision du modèle (overlifting par exemple)
    #Je crois que c'est pertinent mais il faut le faire avec précaution et avoir conscience des risques que cela comporte
    #Il faudrait tester plusieurs méthodes d'undersampling et d'oversampling pour voir si cela a un impact sur la précision du modèle
    #Et il faudra utiliser du class_weight pour rééquilibrer les classes
    df_begnin = df[df[' Label'] == 'BENIGN']
    # Using .isin() for cleaner and more efficient filtering
    df_dos = df[df[' Label'].isin(['DoS GoldenEye', 'Dos Hulk', 'Dos Slowhttptest', 'Dos slowloris', 'DDoS'])]
    df_dos.loc[:, ' Label'] = 'DoS'
    df_portscan = df[df[' Label'] == 'PortScan']
    df_portscan.loc[:, ' Label'] = 'PortScan'
    # Fix similar issues in other filtering operations
    df_infiltration = df[df[' Label'].isin(['Infiltration', 'Heartbleed'])]
    df_infiltration.loc[:, ' Label'] = 'Infiltration'
    df_webattacks = df[df[' Label'].str.startswith('Web Attack')]
    df_webattacks.loc[:, ' Label'] = 'WebAttacks'
    df_bruteforce = df[df[' Label'].isin(['SSH-Patator', 'FTP-Patator'])]
    df_bruteforce.loc[:, ' Label'] = 'Brute Force'
    df_botnet = df[df[' Label'] == 'Bot']
    df_botnet.loc[:, ' Label'] = 'Botnet'
    df_s = pd.concat([df_s, df_begnin.sample(n=None, frac=0.3, random_state=1)])
    df_s = pd.concat([df_s, df_dos, df_portscan, df_infiltration, df_webattacks, df_bruteforce, df_botnet])
    # df_malware = df[df[' Label'] != 'BENIGN']
    # df_s = pd.concat([df_s, df_begnin.sample(n=None, frac=0.3, random_state=1)])
    # df_s = pd.concat([df_s, df_malware])

toPreprocess = []
totalSamples = 0
for key in wordsCount:
    totalSamples += wordsCount[key]
    print(f"Le mot '{key}' apparaît {wordsCount[key]} fois.")
    if wordsCount[key] < 500:
        toPreprocess.append(key)
print(f"Les mots à Oversampler sont: {toPreprocess}")
print(f"Le nombre total d'échantillons est de {totalSamples}")

print("taille de df_s: ", len(df_s))
print("Soit une réduction de : ", len(df_s)/totalSamples*100, "%")
print(df_s[' Label'].value_counts())
df_s.to_csv("MachineLearningCVE\\dataSampled_Grouped.csv", index=False)

"""Certains échantillons sont largement minoritaires (11 occurences de Heartbleed, 36 Infiltration...)
mais ces très faible échantillon ne permettent même pas d'oversampling car il risquerait de créer du bruit plus qu'autre chose.
Les solutions possibles sont:
Jittering (bruit gaussien) :	
    Facile, préserve la structure des données
    Peut être sous-optimal
Bootstrapping dupliquer puis légèrement modifier les données :	    
    Simple et efficace	
    Risque d’overfitting si mal dosé
Class Weights permet de donner plus de poids aux classes minoritaires :	    
    Ne change pas les données	
    Ne corrige pas le déséquilibre dans les features
XGBoost / LightGBM Ces algorithmes sont robustes aux déséquilibres :	
    Optimisé pour les déséquilibres	
    Plus complexe à tuner
"""

""" 
    BENIGN  
    DoS : { ddos, Dos GoldenEye, Dos Hulk, Dos Slowhttptest, Dos slowloris }
    PortScan : { PortScan }
    Infiltration : { Infiltration }
    WebAttacks : { Brute Force -Web, Brute Force -XSS, SQL Injection }
    Brute Force : { SSH-Patator, FTP-Patator }
    Botnet : { Bot }
 """