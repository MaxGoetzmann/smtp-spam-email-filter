import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import utils


def do_kmeans(input_df):
    # Load data
    df = pd.read_csv(input_df)

    # Preprocessing
    # Convert categorical to numerical
    def fit_transform_to_str(field):
        df[field] = LabelEncoder().fit_transform(df[field])

    fit_transform_to_str(utils.FIELD_SENDER)
    utils.repeat_for_subject_and_body(fit_transform_to_str)

    # Separate features and target
    df.drop(utils.FIELD_HASH, inplace=True, axis=1)
    X = df.drop(utils.FIELD_CLASSIFIER_TRUTH, axis=1)
    y = df[utils.FIELD_CLASSIFIER_TRUTH]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Extraction/Reduction with Autoencoder
    input_dim = X_scaled.shape[1]

    autoencoder = Sequential(
        [
            Dense(utils.NN_REDUCTION_DIM, input_shape=(input_dim,), activation="relu"),
            Dense(input_dim, activation="sigmoid"),
        ]
    )

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=utils.CLUSTER_EPOCHS,
        batch_size=utils.CLUSTER_BATCH_SIZE,
        shuffle=True,
    )

    # Extract the encoded features
    encoder = Sequential(autoencoder.layers[:1])
    X_encoded = encoder.predict(X_scaled)

    # Silhouette Analysis
    silhouette_scores = {}
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_encoded)

        silhouette_avg = silhouette_score(X_encoded, clusters)
        silhouette_scores[n_clusters] = silhouette_avg
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

    print("\nSilhouette Scores for each number of clusters:")
    for k, v in silhouette_scores.items():
        print(f"{k} clusters: {v}")


def main():
    do_kmeans(utils.EMAIL_CSV_PATH)


if __name__ == "__main__":
    main()
