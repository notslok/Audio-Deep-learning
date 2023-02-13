import json
import numpy as np
from sklearn.model_selection import train_test_split
DATASET_PATH = "/home/slok/PycharmProjects/pythonProject1/DeepLearningForAudio/gtzan_dataset/Data/data.json" # approx 10 segments per song
import tensorflow as tf

keras = tf.keras
# load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # convert list into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":

    # load data

    inputs, targets = load_data(DATASET_PATH)

    # split data into train and test set

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # build the network achitecture

    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),        # flattening every MFCCs for each track...1st dim->time interval...and 2nd dim-> mfcc val for that interval...0th dim-> represents different segment

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),

        # Output layer
        keras.layers.Dense(10, activation="softmax")   # 10 neurons in o/p layer bcz we have 10 categories to classify from
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # train network
    model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=50,
              batch_size=32)














