import os
import librosa
import math
import json
import numpy as np
DATASET_PATH = "/home/slok/PycharmProjects/pythonProject1/DeepLearningForAudio/gtzan_dataset/Data/genres_original"
JSON_PATH = "/home/slok/PycharmProjects/pythonProject1/DeepLearningForAudio/gtzan_dataset/Data/data.json"

SAMPLE_RATE = 22050
DURATION = 30   # as each rack is 30 sec long
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):   # dividing each audio sample in segment in a way to augment the data

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # calculate number of samples per segment
    number_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    expected_num_mfcc_vectors_per_segment = math.ceil(number_samples_per_segment / hop_length)   # bcz we are calculating mffc at every hop

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):   # i gets us the iteration number

        # ensure that we are not the root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/")    # genre/blues => [...,"genre", "blues"]
            semantic_label = dirpath_components[-1]    # gets the last element of dirpath_components
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = number_samples_per_segment * s # ... s=0 -> 0
                    finish_sample = start_sample + number_samples_per_segment # ... s=0 -> number_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T # taking transpose for ease of operation


                    # store mfcc for segment if it has expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())   # converting numpy to list
                        data["labels"].append(i-1)   # at each iteration i we are in different genre folder except for iteration i=0...which gives us the root dirpath itself so we ignore i=0 case
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)