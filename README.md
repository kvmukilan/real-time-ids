# AI-Powered Real-Time Intrusion Detection System (IDS)

This project implements a lightweight real-time IDS using a deep learning AutoEncoder trained on NSL-KDD data, integrated with `scapy` to monitor live network traffic.

## 🔍 Features
- Unsupervised anomaly detection using AutoEncoder
- Real-time packet sniffing with Scapy
- MSE-based anomaly flagging (no need for labeled attack data)

## 📦 Files
- `train_5feature_autoencoder.py` – Trains and saves the AutoEncoder model
- `realtime_ids.py` – Real-time detection script using Scapy
- `.h5`, `.npy` – Pretrained model, scaler, and threshold

## 📊 Dataset
[NSL-KDD Dataset](https://github.com/defcom17/NSL_KDD)


