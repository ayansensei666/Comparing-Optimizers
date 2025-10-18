import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_model(p_dropout=0.3, activation='relu'):
    return keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation=activation),
        layers.Dropout(p_dropout),
        layers.Dense(128, activation=activation),
        layers.Dropout(p_dropout),
        layers.Dense(10, activation='softmax')
    ])

def load_data(subset_size=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if subset_size and subset_size > 0:
        x_train, y_train = x_train[:subset_size], y_train[:subset_size]
    return (x_train, y_train), (x_test, y_test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    with open("C:\\Users\\aluas\\OneDrive\\Рабочий стол\\TeamZBA\\config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    (x_train, y_train), (x_test, y_test) = load_data(cfg["subset_size"])

    model = build_model(cfg["p_dropout"], cfg["activation"])

    # выбираем оптимайзер
    if cfg["optimizer"] == "sgd":
        opt = keras.optimizers.SGD(learning_rate=cfg["lr"])
    elif cfg["optimizer"] == "momentum":
        opt = keras.optimizers.SGD(learning_rate=cfg["lr"], momentum=0.9)
    elif cfg["optimizer"] == "adam":
        opt = keras.optimizers.Adam(learning_rate=cfg["lr"])
    else:
        raise ValueError("Unknown optimizer")

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        verbose=2
    )

    # сохраняем историю обучения
    os.makedirs("histories", exist_ok=True)
    np.save(f"histories/history_{cfg['optimizer']}.npy", history.history)

    print(f"\n✅ История сохранена: histories/history_{cfg['optimizer']}.npy")

if __name__ == "__main__":
    main()
