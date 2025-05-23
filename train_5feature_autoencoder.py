import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load dataset
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target", "difficulty"
]

data = pd.read_csv("KDDTrain+.txt", names=column_names)
data.drop("difficulty", axis=1, inplace=True)
data["target"] = data["target"].apply(lambda x: 0 if x.strip().lower() == "normal" else 1)

# Keep only 5 simple features
features = ["src_bytes", "protocol_type", "service", "flag", "dst_bytes"]
X = data[features]
y = data["target"]

# Encode categorical
for col in ["protocol_type", "service", "flag"]:
    X[col] = LabelEncoder().fit_transform(X[col])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train only on normal data
X_train = X_scaled[y == 0]

# Build AutoEncoder
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(X_train.shape[1], activation="linear")(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.fit(X_train, X_train, epochs=20, batch_size=128, validation_split=0.1)

# Save model and preprocessing components
autoencoder.save("autoencoder_5feature.h5")
np.save("scaler_mean_5.npy", scaler.mean_)
np.save("scaler_scale_5.npy", scaler.scale_)

# Save threshold
recon = autoencoder.predict(X_train)
mse = np.mean(np.power(X_train - recon, 2), axis=1)
threshold = np.percentile(mse, 95)
np.save("threshold_5.npy", threshold)

print("✅ Saved model, scaler, and threshold.")
