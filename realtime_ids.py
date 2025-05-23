import time
import numpy as np
from scapy.all import sniff
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and threshold
model = load_model("autoencoder_5feature.h5", compile=False)
threshold = float(np.load("threshold_5.npy"))

# Load scaler
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean_5.npy")
scaler.scale_ = np.load("scaler_scale_5.npy")

# Encode protocol_type, service, and flag manually (map should match training)
protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}  # adjust if needed
service_map = {"http": 0, "ftp": 1, "domain": 2, "private": 3, "other": 4}  # customize
flag_map = {"SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "OTH": 4}  # customize

# Function to extract simplified features
def extract_features(pkt):
    try:
        length = len(pkt)
        proto = pkt.payload.name.lower()
        is_tcp = 1 if "tcp" in proto else 0
        is_udp = 1 if "udp" in proto else 0
        is_icmp = 1 if "icmp" in proto else 0

        # Use protocol name and TCP flags to simulate mappings
        protocol = protocol_map.get(proto, 0)
        service = service_map.get(pkt.summary().split()[0].lower(), 0)
        flag = 0
        if "TCP" in pkt:
            tcp_flags = str(pkt["TCP"].flags)
            if "S" in tcp_flags and "A" not in tcp_flags:
                flag = flag_map.get("S0", 1)
            elif "R" in tcp_flags:
                flag = flag_map.get("REJ", 2)
            else:
                flag = flag_map.get("SF", 0)
        else:
            flag = flag_map.get("OTH", 4)

        src_bytes = pkt.len if hasattr(pkt, "len") else length
        dst_bytes = 0  # Cannot infer easily in one direction

        feature_vector = np.array([[src_bytes, protocol, service, flag, dst_bytes]])
        return feature_vector
    except Exception as e:
        print("Feature extraction error:", e)
        return None

# Real-time detection function
def detect_anomaly(pkt):
    features = extract_features(pkt)
    if features is not None:
        try:
            scaled = scaler.transform(features)
            recon = model.predict(scaled)
            mse = np.mean(np.power(scaled - recon, 2))

            if mse > threshold:
                print(f"[⚠️ Anomaly] MSE={mse:.4f} — {pkt.summary()}")
            else:
                print(f"[✅ Normal] MSE={mse:.4f}")

        except Exception as e:
            print("Prediction error:", e)

# Start sniffing
print("🚀 Real-time IDS started. Listening on default interface...")
sniff(prn=detect_anomaly, store=0)
