import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

keras = tf.keras
DATA_PATH = "/home/slok/PycharmProjects/pythonProject1/DeepLearningForAudio/gtzan_dataset/Data/data.json"


def load_data(data_path):
    """
        loads training dataset from json file.

        :param data_path: Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    label_name = np.array(data["mapping"])

    return X, y, label_name


def prepare_datasets(test_size, validation_size):
    # load data
    X, y, label_name = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # extra dimension not needed for rnn
    # # we have at this point... 3d array -> (130, 13, 1)
    # X_train = X_train[..., np.newaxis]  # now it becomes 4d array -> (num_samples, 130, 13, 1)
    # X_validation = X_validation[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_name


def build_model(input_shape):
    # create rnn-lstm model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_state=True))    # ... (nums of steps, input_shape, seq2seq or seq2vec)
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y, label_name):
    # X -> (130, 3, 1)
    # X = X[np.newaxis, ...]  # now ... X -> (1, 130, 3, 1)

    # prediction = [[0.1, 0.3, ...]]
    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # ---> [index]
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
    print("Expected label: {}, Predicted label: {}".format(label_name[y], label_name[np.squeeze(predicted_index)]))


if __name__ == "__main__":
    #  create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, label_name = prepare_datasets(0.25, 0.2)

    #  build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2])  # ... (130, 13)
    model = build_model(input_shape)

    #  compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    #  train the CNN
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30,
              verbose=1)  # verbose=1 will show the animated progress bar according o model.fit() documentation

    #  Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on the test set is: {}".format(test_accuracy))

    #  make prediction on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y, label_name)
