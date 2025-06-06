# exemple = dict({"dst_port": 443, "flow_duration": 121.82646179199219, "flow_byts_s": 8.0278125590633, "bwd_pkts_s": 0.049250383797934345, "tot_fwd_pkts": 6, "tot_bwd_pkts": 6, "totlen_fwd_pkts": 495, "fwd_pkt_len_max": 95, "fwd_pkt_len_mean": 82.5, "bwd_pkt_len_max": 91, "bwd_pkt_len_min": 66, "bwd_pkt_len_mean": 80.5, "bwd_pkt_len_std": 11.236102527122116, "pkt_len_max": 95, "pkt_len_mean": 81.5, "pkt_len_std": 12.257650672131263, "pkt_len_var": 150.25, "fwd_act_data_pkts": 3, "flow_iat_max": 61.119901180267334, "flow_iat_std": 23.316847235599933, "fwd_iat_tot": 60.51107597351074, "fwd_iat_max": 60.12160921096802, "fwd_iat_std": 24.01016735059795, "psh_flag_cnt": 6, "ack_flag_cnt": 12, "pkt_size_avg": 81.5, "init_fwd_win_byts": 470, "fwd_seg_size_avg": 82.5, "bwd_seg_size_avg": 80.5, "subflow_fwd_pkts": 6, "subflow_fwd_byts": 495, "subflow_bwd_byts": 483})
#
# original_ordered_features = [
#         "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min" , "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
#         ]
# cic_to_model = {
#     "dst_port": "Destination Port",
#     "flow_duration": "Flow Duration",
#     "tot_fwd_pkts": "Total Fwd Packets",
#     "tot_bwd_pkts": "Total Backward Packets",
#     "totlen_fwd_pkts": "Total Length of Fwd Packets",
#     "totlen_bwd_pkts": "Total Length of Bwd Packets",
#     "fwd_pkt_len_max": "Fwd Packet Length Max",
#     "fwd_pkt_len_min": "Fwd Packet Length Min",
#     "fwd_pkt_len_mean": "Fwd Packet Length Mean",
#     "fwd_pkt_len_std": "Fwd Packet Length Std",
#     "bwd_pkt_len_max": "Bwd Packet Length Max",
#     "bwd_pkt_len_min": "Bwd Packet Length Min",
#     "bwd_pkt_len_mean": "Bwd Packet Length Mean",
#     "bwd_pkt_len_std": "Bwd Packet Length Std",
#     "flow_byts_s": "Flow Bytes/s",
#     "flow_pkts_s": "Flow Packets/s",
#     "flow_iat_mean": "Flow IAT Mean",
#     "flow_iat_std": "Flow IAT Std",
#     "flow_iat_max": "Flow IAT Max",
#     "flow_iat_min": "Flow IAT Min",
#     "fwd_iat_tot": "Fwd IAT Total",
#     "fwd_iat_mean": "Fwd IAT Mean",
#     "fwd_iat_std": "Fwd IAT Std",
#     "fwd_iat_max": "Fwd IAT Max",
#     "fwd_iat_min": "Fwd IAT Min",
#     "bwd_iat_tot": "Bwd IAT Total",
#     "bwd_iat_mean": "Bwd IAT Mean",
#     "bwd_iat_std": "Bwd IAT Std",
#     "bwd_iat_max": "Bwd IAT Max",
#     "bwd_iat_min": "Bwd IAT Min",
#     "fwd_psh_flags": "Fwd PSH Flags",
#     "bwd_psh_flags": "Bwd PSH Flags",
#     "fwd_urg_flags": "Fwd URG Flags",
#     "bwd_urg_flags": "Bwd URG Flags",
#     "fwd_header_len": "Fwd Header Length",
#     "bwd_header_len": "Bwd Header Length",
#     "fwd_pkts_s": "Fwd Packets/s",
#     "bwd_pkts_s": "Bwd Packets/s",
#     "pkt_len_min": "Min Packet Length",
#     "pkt_len_max": "Max Packet Length",
#     "pkt_len_mean": "Packet Length Mean",
#     "pkt_len_std": "Packet Length Std",
#     "pkt_len_var": "Packet Length Variance",
#     "fin_flag_cnt": "FIN Flag Count",
#     "syn_flag_cnt": "SYN Flag Count",
#     "rst_flag_cnt": "RST Flag Count",
#     "psh_flag_cnt": "PSH Flag Count",
#     "ack_flag_cnt": "ACK Flag Count",
#     "urg_flag_cnt": "URG Flag Count",
#     # "cwe_flag_cnt": "CWE Flag Count",  # Il faut l'ajouter manuellement
#     "ece_flag_cnt": "ECE Flag Count",  # Remplace CWE Flag Count
#     "down_up_ratio": "Down/Up Ratio",
#     "pkt_size_avg": "Average Packet Size",
#     "fwd_seg_size_avg": "Avg Fwd Segment Size",
#     "bwd_seg_size_avg": "Avg Bwd Segment Size",
#     # "fwd_header_len": "Fwd Header Length.1",  # Dupliqué il faut l'ajouter manuellement
#     "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
#     "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
#     "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
#     "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
#     "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
#     "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
#     "subflow_fwd_pkts": "Subflow Fwd Packets",
#     "subflow_fwd_byts": "Subflow Fwd Bytes",
#     "subflow_bwd_pkts": "Subflow Bwd Packets",
#     "subflow_bwd_byts": "Subflow Bwd Bytes",
#     "init_fwd_win_byts": "Init_Win_bytes_forward",
#     "init_bwd_win_byts": "Init_Win_bytes_backward",
#     "fwd_act_data_pkts": "act_data_pkt_fwd",
#     "fwd_seg_size_min": "min_seg_size_forward",
#     "active_mean": "Active Mean",
#     "active_std": "Active Std",
#     "active_max": "Active Max",
#     "active_min": "Active Min",
#     "idle_mean": "Idle Mean",
#     "idle_std": "Idle Std",
#     "idle_max": "Idle Max",
#     "idle_min": "Idle Min",
# }
# #
# # keys = list(exemple.keys())
# # for i in range(0, len(exemple.keys())):
# #     key = keys[i]
# #     indexes = list(cic_to_model.keys())
# #     prev_key_index = indexes.index(keys[i-1]) if i-1 >= 0 else 0
# #     key_index = indexes.index(key)
# #     if i+1 >= len(indexes):
# #         next_key_index = len(indexes) - 1
# #     else:
# #         next_key_index = indexes.index(keys[i+1])
# #     print("prev_key_index:", prev_key_index, "key_index:", key_index, "next_key_index:", next_key_index)
# #     if key_index < prev_key_index or key_index > next_key_index:
# #         print("Key out of order:", key)
# #     # if key_index < prev_index:
# #     #     print("Key out of order:", key)
# #
# #     # if key not in cic_to_model:
# #     #     print("Key not found in cic_to_model:", key)
#
# print(len(exemple.keys()))
# model_data = {}
# for i in range(0, len(exemple.keys())):
#     key = list(exemple.keys())[i]
#     model_name = cic_to_model.get(key)
#     if model_name is not None:
#         model_data[model_name] = exemple[key]
#     else:
#         print("Key not found in cic_to_model:", key)
#
# # print(model_data)
# for k, v in model_data.items():
#     if k not in original_ordered_features:
#         print("Key not found in original_ordered_features:", k)
# # model_data_ordered = {k: model_data[k] for k in original_ordered_features} # if k in model_data}
# model_data_ordered = {}
# key_founded = 0
# for k in original_ordered_features:
#     if k in model_data:
#         model_data_ordered[k] = model_data[k]
#         key_founded += 1
#     else:
#         print("Key not found in model_data:", k)
# print(model_data_ordered)
#
# # DataFrame dans le bon ordre
# df = pd.DataFrame([model_data_ordered])
#

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

X = joblib.load('../X_fs.pkl')
# X_fs = pd.DataFrame(X_fs)
y = pd.DataFrame(joblib.load('../y.pkl'))
pipeline, le = joblib.load("../stacking_model.joblib")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

y_train = y_train.values.ravel()  # Convert to 1D array
y_test = y_test.values.ravel()  # Convert to 1D array
y_pred = pipeline.predict([X_train[0]])
print("Predicted label:", y_pred[0], "True label:", y_train[0])
y_pred_test = pipeline.predict(X_test[0].reshape(1, -1))
print("Predicted label for test data:", y_pred_test[0], "True label:", y_test[0])
