import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Load and prepare data
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

# Step 2: Preprocess features
X = data.drop("target", axis=1)
y = data["target"]
cat_cols = X.select_dtypes(include=["object"]).columns
X[cat_cols] = X[cat_cols].apply(LabelEncoder().fit_transform)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split - Train on normal only
X_train = X_scaled[y == 0]
X_test = X_scaled
y_test = y

# Step 4: Build improved AutoEncoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation="relu")(input_layer)
encoder = Dense(64, activation="relu")(encoder)
encoder = Dense(16, activation="relu")(encoder)

decoder = Dense(64, activation="relu")(encoder)
decoder = Dense(128, activation="relu")(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Step 5: Train the model
autoencoder.fit(X_train, X_train, epochs=20, batch_size=128, validation_split=0.1)

# Step 6: Evaluate on full test data
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

""" Step 7: Visualize MSE distribution (optional)
plt.figure(figsize=(10, 6))
plt.hist(mse[y == 0], bins=100, alpha=0.6, label='Normal')
plt.hist(mse[y == 1], bins=100, alpha=0.6, label='Anomaly')
plt.xlim(0, 100)  # Zoom in
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency")
plt.title("MSE Distribution - AutoEncoder IDS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""""

# Step 8: Set threshold and evaluate
threshold = np.percentile(mse[y == 0], 90)
print(f"\nThreshold (90% of normal): {threshold:.4f}")

y_pred = (mse > threshold).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))


autoencoder.save("autoencoder_ids_model.h5")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
np.save("threshold.npy", threshold)

