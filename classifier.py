import numpy as np
import pandas as pd
import numpy.typing as npt
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import itertools

# defining constants
TEST_SIZE: np.float_ = 0.3
RANDOM_STATE: np.int_ = 42

# importing and handling data
data: pd.DataFrame = pd.read_csv("data/input/data_tp1", header=None).to_numpy()

def MNIST_MLP(data: npt.NDArray[np.int_], hidden_layer_size: np.int_, batch_size: np.int_, learning_rate: np.float_) -> dict:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(784,)),
        tf.keras.layers.Dense(units=hidden_layer_size, activation="sigmoid"),
        tf.keras.layers.Dense(units=10, activation="softmax")
    ])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    epochs: np.int_ = 10

    input_data: npt.NDArray[np.int_] = data[:, 1:]
    input_data = input_data / 255
    labels: npt.NDArray[np.int_] = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

    y_pred: npt.NDArray[np.int_] = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = y_pred.argmax(axis=-1)

    accuracy_score: np.float_ = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision_score: npt.NDArray[np.float_] = sklearn.metrics.precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_score: npt.NDArray[np.float_] = sklearn.metrics.recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_score: npt.NDArray[np.float_] = sklearn.metrics.f1_score(y_test, y_pred, average=None)
    confusion_matrix: npt.NDArray[np.int_] = sklearn.metrics.confusion_matrix(y_test, y_pred)

    run_info: dict = {
        "accuracy_score": accuracy_score,
        "precision_score": precision_score.tolist(),
        "recall_score": recall_score.tolist(),
        "f1_score": f1_score.tolist(),
        "confusion_matrix": confusion_matrix.tolist(),
        "random_state_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "hidden_layer_size": hidden_layer_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": model_history.history
    }

    return run_info

hidden_layer_sizes: list[int] = [25, 50, 100]
batch_sizes: list[int] = [1, 10, 50, 3500]
lerning_rates: list[float] = [0.5, 1.0, 10.0]

configurations: list = list(itertools.product(hidden_layer_sizes, batch_sizes, lerning_rates))

run_infos: list[dict] = [MNIST_MLP(data=data, hidden_layer_size=a, batch_size=b, learning_rate=c) for a, b, c in configurations]

with open("data/results.json", "a") as file:
    json.dump([run_info for run_info in run_infos], file, indent=4)
