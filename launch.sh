#!/bin/bash

# === Vérification explicite des droits sudo ===
echo "🔐 Ce script nécessite les droits administrateur pour lancer CICFlowMeter."
sudo -v || { echo "❌ Échec de l'authentification sudo."; exit 1; }

# === Gestion du paramètre -v ===
VERBOSE=""
if [[ "$1" == "-v" ]]; then
    VERBOSE="-v"
    echo "⚡ Mode verbeux activé pour CICFlowMeter"
fi

# === Fonction d'arrêt propre ===
function cleanup {
    echo -e "\n🛑 Arrêt du script demandé. On termine les processus..."

    if kill -0 "$PID_CIC" 2>/dev/null; then
        echo " - Fermeture de CICFlowMeter (PID $PID_CIC)"
        sudo kill "$PID_CIC"
    fi

    if kill -0 "$PID_API" 2>/dev/null; then
        echo " - Fermeture de FastAPI (PID $PID_API)"
        kill "$PID_API"
    fi

    echo "✅ Tous les processus ont été arrêtés."
    exit 0
}

trap cleanup SIGINT SIGTERM

# === Lancer CICFlowMeter ===
echo "[*] Lancement de CICFlowMeter..."
source wrapper/cicflowmeter/.venv/bin/activate
sudo wrapper/cicflowmeter/.venv/bin/cicflowmeter -i wlo1 -u http://localhost:8000/flow $VERBOSE \
--fields pkt_len_var,fwd_pkt_len_max,subflow_fwd_byts,psh_flag_cnt,bwd_pkt_len_std,totlen_fwd_pkts,fwd_act_data_pkts,dst_port,bwd_pkts_s,fwd_iat_max,bwd_pkt_len_mean,bwd_seg_size_avg,pkt_len_std,pkt_size_avg,pkt_len_mean,ack_flag_cnt,pkt_len_max,bwd_pkt_len_max,fwd_seg_size_avg,init_fwd_win_byts,bwd_pkt_len_min,fwd_pkt_len_mean,flow_byts_s,tot_bwd_pkts,tot_fwd_pkts,flow_duration,subflow_fwd_pkts,flow_iat_max,fwd_iat_std,subflow_bwd_byts,fwd_iat_tot,flow_iat_std \
&
PID_CIC=$!
deactivate
echo "[+] CICFlowMeter lancé (PID $PID_CIC)"

# === Lancer FastAPI ===
echo "[*] Lancement de l'API FastAPI..."
source CNN/venv/bin/activate
fastapi run wrapper/api.py &
PID_API=$!
deactivate
echo "[+] FastAPI lancé (PID $PID_API)"

# === Infos utilisateur ===
echo
echo "🎉 Les deux services sont en cours :"
echo " - CICFlowMeter : PID $PID_CIC"
echo " - FastAPI      : PID $PID_API"
echo
echo "⏳ Appuie sur CTRL+C pour tout arrêter proprement."
echo

# === Garder le script vivant ===
while true; do
    sleep 1
done

