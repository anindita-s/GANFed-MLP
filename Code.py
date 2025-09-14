import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

df = pd.read_csv("IoT-filtered.csv")
print("Initial Class Distribution:\n", df['Label'].value_counts())

X = df.drop(columns=['Label', 'Attack Name', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'])
y = df['Label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
def create_mlp(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_dim = X_train_resampled.shape[1]

mlp_model = create_mlp(input_dim)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
weights_dict = dict(enumerate(weights))
mlp_model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, class_weight=weights_dict, verbose=1)


def build_generator(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(input_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

gen = build_generator(input_dim)
disc = build_discriminator(input_dim)
disc.trainable = False

gan_input = keras.Input(shape=(input_dim,))
gan_output = disc(gen(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

real_data = X_train_resampled[y_train_resampled == 1]
for epoch in range(0, 5001, 1):
    idx = np.random.randint(0, real_data.shape[0], 64)
    real_samples = real_data[idx]
    noise = np.random.normal(0, 1, (64, input_dim))
    fake_samples = gen.predict(noise, verbose=0)

    X_disc = np.concatenate([real_samples, fake_samples])
    y_disc = np.concatenate([np.ones((64, 1)), np.zeros((64, 1))])

    disc.trainable = True
    d_loss = disc.train_on_batch(X_disc, y_disc)

    noise = np.random.normal(0, 1, (64, input_dim))
    y_gen = np.ones((64, 1))
    disc.trainable = False
    g_loss = gan.train_on_batch(noise, y_gen)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")


noise = np.random.normal(0, 1, (300000, input_dim))
synthetic_samples = gen.predict(noise, verbose=0)
predicted_labels = mlp_model.predict(synthetic_samples, verbose=0)
accepted_idx = np.where(predicted_labels > 0.8)[0]
X_aug = synthetic_samples[accepted_idx]
y_aug = np.ones(len(X_aug))
print("Accepted GAN samples after pseudo-labeling:", len(X_aug))

X_final = np.vstack([X_train_resampled, X_aug])
y_final = np.hstack([y_train_resampled, y_aug])
X_labeled = X_final
y_labeled = y_final
X_unlabeled = synthetic_samples


def create_global_model():
    return create_mlp(input_dim)

def aggregate_models(client_models):
    global_weights = [np.zeros_like(w) for w in client_models[0].get_weights()]
    for model in client_models:
        for i, w in enumerate(model.get_weights()):
            global_weights[i] += w
    global_weights = [w / len(client_models) for w in global_weights]
    return global_weights

def train_client_model(X_labeled_client, y_labeled_client, X_unlabeled, epochs):
    model = create_mlp(input_dim)
    history1 = model.fit(X_labeled_client, y_labeled_client, epochs=epochs, batch_size=32, verbose=0)
    pseudo_labels = model.predict(X_unlabeled, verbose=0)
    pseudo_idx = np.where(pseudo_labels > 0.8)[0]
    X_pseudo = X_unlabeled[pseudo_idx]
    y_pseudo = np.ones(len(X_pseudo))

    X_combined = np.vstack([X_labeled_client, X_pseudo])
    y_combined = np.hstack([y_labeled_client, y_pseudo])
    history2 = model.fit(X_combined, y_combined, epochs=epochs, batch_size=32, verbose=0)

    return model, history1.history['accuracy'], history2.history['accuracy']

def train_client_model(X_labeled_client, y_labeled_client, X_unlabeled, epochs):
    model = create_mlp(input_dim)
    history1 = model.fit(X_labeled_client, y_labeled_client, epochs=epochs, batch_size=32, verbose=0)
    pseudo_labels = model.predict(X_unlabeled, verbose=0)
    pseudo_idx = np.where(pseudo_labels > 0.8)[0]
    X_pseudo = X_unlabeled[pseudo_idx]
    y_pseudo = np.ones(len(X_pseudo))

    X_combined = np.vstack([X_labeled_client, X_pseudo])
    y_combined = np.hstack([y_labeled_client, y_pseudo])
    history2 = model.fit(X_combined, y_combined, epochs=epochs, batch_size=32, verbose=0)

    return model, history1.history['accuracy'], history2.history['accuracy']

def federated_learning_experiment(X_labeled, y_labeled, X_unlabeled, n_clients_list, epoch_list):
    fl_results = {}
    accuracy_tracking = {}

    for n_clients in n_clients_list:
        client_data_splits = np.array_split(X_labeled, n_clients)
        client_label_splits = np.array_split(y_labeled, n_clients)

        for epochs in epoch_list:
            client_models = []
            train_acc_all, val_acc_all = [], []

            for X_client, y_client in zip(client_data_splits, client_label_splits):
                client_model, acc1, acc2 = train_client_model(X_client, y_client, X_unlabeled, epochs)
                client_models.append(client_model)
                train_acc_all.append(acc1)
                val_acc_all.append(acc2)

            global_model = create_global_model()
            global_weights = aggregate_models(client_models)
            global_model.set_weights(global_weights)

            loss, accuracy = global_model.evaluate(X_test, y_test, verbose=0)
            fl_results[(n_clients, epochs)] = accuracy
            accuracy_tracking[(n_clients, epochs)] = (train_acc_all, val_acc_all)

            print(f"Clients: {n_clients}, Epochs: {epochs}, Accuracy: {accuracy:.4f}")
            global_model.save("/content/drive/MyDrive/model/global_federated_model.h5")
            print("✅ Saved: 'global_federated_model.h5'")

    return fl_results, accuracy_tracking
