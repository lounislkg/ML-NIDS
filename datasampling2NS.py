import pandas as pd
import numpy as np

file_path = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

# Initialize variables
wordsCount = dict()
df_s = pd.DataFrame()

# Define malware categories
malware_labels = [
    "DDoS", "DoS GoldenEye", "Dos Hulk", "Dos Slowhttptest", "Dos slowloris",
    "PortScan", "Infiltration", "Web Attack - Brute Force", "Web Attack - XSS",
    "Web Attack - Sql Injection", "SSH-Patator", "FTP-Patator", "Bot"
]

# Process each file
for file in file_path:
    df = pd.read_csv("archive\\" + file)
    
    # Map labels to BENIGN or MALWARE
    df[' Label'] = df[' Label'].apply(lambda x: "MALWARE" if x in malware_labels else "BENIGN")
    
    # Count occurrences of each label
    counts = df[" Label"].value_counts()
    for label in counts.index:
        if label not in wordsCount:
            wordsCount[label] = int(counts[label])
        else:
            wordsCount[label] += int(counts[label])
    
    # Sample 30% of BENIGN data and keep all MALWARE data
    df_benign = df[df[' Label'] == 'BENIGN']
    df_malware = df[df[' Label'] == 'MALWARE']
    df_s = pd.concat([df_s, df_benign.sample(frac=0.3, random_state=1), df_malware])

# Display statistics
totalSamples = sum(wordsCount.values())
print(f"Le nombre total d'échantillons est de {totalSamples}")
print("taille de df_s: ", len(df_s))
print("Soit une réduction de : ", len(df_s) / totalSamples * 100, "%")
print(df_s[' Label'].value_counts())

# Save the processed dataset
df_s.to_csv("archive\\dataSampled2.csv", index=False)