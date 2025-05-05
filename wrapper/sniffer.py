import subprocess
import time


def capture_pcap(duration=5, output_file="capture.pcap", interface="eth0"):
    try:
        print(f"[+] Démarrage de la capture sur {interface} pendant {duration}s...")

        result = subprocess.run(
            [
                "tcpdump", "-i", interface,
                "-w", output_file,
                "-G", str(duration),
                "-W", "1",
                "-nn", "-q"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # decode en str directement
        )

        if result.returncode == 0:
            print("[+] Capture terminée avec succès.")
        else:
            print("[!] Erreur pendant la capture TCPDUMP.")

        # stdout est souvent vide pour tcpdump, stderr contient les logs
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)

    except FileNotFoundError:
        print("[X] tcpdump n'est pas installé.")
    except PermissionError:
        print("[X] Permission refusée pour lancer tcpdump (utiliser sudo ?)")
    except Exception as e:
        print(f"[X] Erreur inconnue : {e}")

capture_pcap(duration=5, output_file="capture.pcap", interface="wlo1")


# def run_cicflowmeter(pcap_file, output_csv="flows.csv"):
#     subprocess.run([
#         "java", "-jar", "CICFlowMeter-4.0.jar",
#         "-f", pcap_file,
#         "-c", output_csv
#     ])
#


