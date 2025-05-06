import pandas as pd

original_ordered_features = [
    "Packet Length Variance",
    "Fwd Packet Length Max",
    "Subflow Fwd Bytes",
    "PSH Flag Count",
    "Bwd Packet Length Std",
    "Total Length of Fwd Packets",
    "act_data_pkt_fwd",
    "Destination Port",
    "Bwd Packets/s",
    "Fwd IAT Max",
    "Bwd Packet Length Mean",
    "Avg Bwd Segment Size",
    "Packet Length Std",
    "Average Packet Size",
    "Packet Length Mean",
    "ACK Flag Count",
    "Max Packet Length",
    "Bwd Packet Length Max",
    "Avg Fwd Segment Size",
    "Init_Win_bytes_forward",
    "Bwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Flow Bytes/s",
    "Total Backward Packets",
    "Total Fwd Packets",
    "Flow Duration",
    "Subflow Fwd Packets",
    "Flow IAT Max",
    "Fwd IAT Std",
    "Subflow Bwd Bytes",
    "Fwd IAT Total",
    "Flow IAT Std"
]

cic_to_model = {
    "dst_port": "Destination Port",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwe_flag_cnt": "CWE Flag Count",  # Il faut l'ajouter manuellement
    "ece_flag_cnt": "ECE Flag Count",  # Remplace CWE Flag Count
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    # "fwd_header_len": "Fwd Header Length.1",  # Dupliqué il faut l'ajouter manuellement
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}
#
# cicflow_data = {
#     "dst_port": 443,
#     "flow_duration": 100000,
#     "tot_fwd_pkts": 10,
#     "tot_bwd_pkts": 12,
#     "totlen_fwd_pkts": 1200.0,
#     "totlen_bwd_pkts": 1500.0,
#     "fwd_pkt_len_max": 300.0,
#     "fwd_pkt_len_min": 50.0,
#     "fwd_pkt_len_mean": 120.0,
#     "fwd_pkt_len_std": 40.0,
#     "bwd_pkt_len_max": 400.0,
#     "bwd_pkt_len_min": 60.0,
#     "bwd_pkt_len_mean": 125.0,
#     "bwd_pkt_len_std": 50.0,
#     "flow_byts_s": 25000.0,
#     "flow_pkts_s": 220.0,
#     "flow_iat_mean": 1000.0,
#     "flow_iat_std": 300.0,
#     "flow_iat_max": 5000.0,
#     "flow_iat_min": 100.0,
#     "fwd_iat_tot": 5000.0,
#     "fwd_iat_mean": 500.0,
#     "fwd_iat_std": 150.0,
#     "fwd_iat_max": 2000.0,
#     "fwd_iat_min": 100.0,
#     "bwd_iat_tot": 6000.0,
#     "bwd_iat_mean": 600.0,
#     "bwd_iat_std": 180.0,
#     "bwd_iat_max": 2500.0,
#     "bwd_iat_min": 120.0,
#     "fwd_psh_flags": 0,
#     "bwd_psh_flags": 1,
#     "fwd_urg_flags": 0,
#     "bwd_urg_flags": 0,
#     "fwd_header_len": 160,
#     "bwd_header_len": 180,
#     "fwd_pkts_s": 100.0,
#     "bwd_pkts_s": 120.0,
#     "pkt_len_min": 50.0,
#     "pkt_len_max": 400.0,
#     "pkt_len_mean": 130.0,
#     "pkt_len_std": 45.0,
#     "pkt_len_var": 2025.0,
#     "fin_flag_cnt": 0,
#     "syn_flag_cnt": 1,
#     "rst_flag_cnt": 0,
#     "psh_flag_cnt": 1,
#     "ack_flag_cnt": 10,
#     "urg_flag_cnt": 0,
#     "ece_flag_cnt": 0,
#     "down_up_ratio": 1.2,
#     "pkt_size_avg": 140.0,
#     "fwd_seg_size_avg": 125.0,
#     "bwd_seg_size_avg": 130.0,
#     "fwd_byts_b_avg": 0.0,
#     "fwd_pkts_b_avg": 0.0,
#     "fwd_blk_rate_avg": 0.0,
#     "bwd_byts_b_avg": 0.0,
#     "bwd_pkts_b_avg": 0.0,
#     "bwd_blk_rate_avg": 0.0,
#     "subflow_fwd_pkts": 10,
#     "subflow_fwd_byts": 1200,
#     "subflow_bwd_pkts": 12,
#     "subflow_bwd_byts": 1500,
#     "init_fwd_win_byts": 65535,
#     "init_bwd_win_byts": 65535,
#     "fwd_act_data_pkts": 8,
#     "fwd_seg_size_min": 20,
#     "active_mean": 20000.0,
#     "active_std": 1000.0,
#     "active_max": 30000.0,
#     "active_min": 15000.0,
#     "idle_mean": 50000.0,
#     "idle_std": 2000.0,
#     "idle_max": 80000.0,
#     "idle_min": 30000.0,
# }


# def reorder_features(cicflow_data: dict) -> pd.DataFrame:
#     # Remap les noms
#     model_data = {
#         model_name: cicflow_data[cic_name] for cic_name, model_name in cic_to_model.items() if model_name in filtered_ordered_features and cic_name in cicflow_data
#     }
#     # DataFrame dans le bon ordre
#     df = pd.DataFrame([model_data])[filtered_ordered_features]
#     return df

# if __name__ == "__main__":
#     # Exemple d'utilisation
#     df = reorder_features(cicflow_data)
#     print(df.head())
#     print(df.shape)
def reorder_features(cicflow_data: dict) -> pd.DataFrame:
    model_data = {}
    for i in range(0, len(cicflow_data.keys())):
        key = list(cicflow_data.keys())[i]
        model_name = cic_to_model.get(key)
        if model_name is not None:
            model_data[model_name] = cicflow_data[key]
        else:
            print("Key not found in cic_to_model:", key)

    # model_data_ordered = {k: model_data[k] for k in original_ordered_features if k in model_data}
    model_data_ordered = {}
    key_founded = 0
    for k in original_ordered_features:
        if k in model_data:
            model_data_ordered[k] = model_data[k]
            key_founded += 1
        else:
            print("Key not found in model_data:", k)
    print(model_data_ordered)
    print("length model_data_ordered:", len(model_data_ordered.keys()))
    # DataFrame dans le bon ordre
    df = pd.DataFrame([model_data_ordered])
    return df
