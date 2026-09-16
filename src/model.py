from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, RandomRotation, RandomTranslation, RandomZoom, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_model(input_shape=(32, 32, 3), num_classes=43):
    """Build a CNN model for GTSRB traffic-sign classification."""
    model = Sequential([
        Input(shape=input_shape),
        RandomRotation(0.028),
        RandomTranslation(0.1, 0.1),
        RandomZoom(0.1),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn = build_model()
    cnn.summary()
