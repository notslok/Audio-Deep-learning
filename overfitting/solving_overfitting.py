import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt

# path to json file that stores MFCCs and genre labels for each proposed segment
DATA_PATH = "/home/slok/PycharmProjects/pythonProject1/DeepLearningForAudio/gtzan_dataset/Data/data.json"

def load_data(data_path):
    """ Loads training dataset from json file
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return Y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert list to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data Successfully loaded!")

    return X, y


def plot_history(history):
    fig, axs = plt.subplot(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title["Accuracy eval"]

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title["Error eval"]


if __name__ == "__main__":
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),  #  flattening every MFCCs for each track...1st dim->time interval...and 2nd dim-> mfcc val for that interval...0th dim-> represents different segment

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)

    # plot the accuracy and error over epochs
    plot_history(history)