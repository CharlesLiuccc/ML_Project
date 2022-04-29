import os

import jams
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from keras.utils.np_utils import to_categorical


class DataPreprocess:
    def __init__(self):
        # the path where the data be loaded
        path = "GuitarSet/"
        self.audio_path = path + "audio/audio_mic/"
        self.anno_path = path + "annotation/"

        # labeling parameters
        self.label_IDs = []
        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2  # for open/closed

        # result of preprocessed data
        self.output = {}

        # CQT parameters
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        self.cqt_n_bits = 192
        self.cqt_bins_per_octave = 24

        self.hop_length = 4096

        # save file path
        self.save_path = "CQT/"

    def preprocess_data(self, filename):
        file_audio = self.audio_path + filename + "_mic.wav"
        file_anno = self.anno_path + filename + ".jams"
        jam = jams.load(file_anno)
        # self.sr_original, data = wavfile.read(file_audio)
        data, self.sr_original = librosa.load(file_audio, sr=None, mono=True)
        self.sr_curr = self.sr_original

        # preprocess audio, store in output dict
        self.output["audio"] = np.swapaxes(self.preprocess_audio(data), 0, 1)
        # print("----audio: \n")
        # print(self.output["audio"])

        # construct labels
        frame_indices = range(len(self.output["audio"]))
        times = librosa.frames_to_time(frame_indices, sr=self.sr_curr, hop_length=self.hop_length)

        # loop over all strings and sample annotations
        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch with fret nums
            for i in frame_indices:
                if not string_label_samples[i]:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(
                        round(string_label_samples[i][0]) - self.string_midi_pitches[string_num])
            labels.append([string_label_samples])

        labels = np.array(labels)
        # remove extra dimension
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels, 0, 1)

        # clean labels
        labels = self.clean_labels(labels)

        # store and return
        self.output["labels"] = labels
        # print("----labels: \n")
        # print(labels)
        return len(labels)

    # modify the labels
    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n

    def categorical(self, label):
        return to_categorical(label, self.num_classes)

    def clean_label(self, label):
        return self.categorical([self.correct_numbering(n) for n in label])

    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])

    # preprocess audio
    def preprocess_audio(self, data):
        data = data.astype(float)
        data = librosa.util.normalize(data)
        # data = librosa.resample(data, self.sr_original, self.sr_downs)
        # CQT
        data = np.abs(librosa.cqt(data,
                                  sr=self.sr_curr,
                                  hop_length=self.hop_length,
                                  n_bins=self.cqt_n_bits,
                                  bins_per_octave=self.cqt_bins_per_octave))
        return data

    def save_data(self, filename):
        dataframe = pd.DataFrame(self.label_IDs)
        dataframe.to_csv("CQT/id.csv", index=False, index_label=False, header=False, mode='a')
        np.savez(filename, **self.output)

    def get_nth_filename(self, n):
        filenames = np.sort(np.array(os.listdir(self.anno_path)))
        # filenames = filter(lambda x: x[-5:] == ".jams", filenames)
        return filenames[n][:-5]

    # preprocess and save nth file
    def preprocess_nth_file(self, n):
        filename = self.get_nth_filename(n)
        num_frames = self.preprocess_data(filename)
        print(str(n) + " done: " + filename + ", " + str(num_frames) + "frames")
        save_path = self.save_path

        # save frames id in id.csv
        for i in range(0, num_frames):
            self.label_IDs.append(filename + "_" + str(i))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_data(save_path + filename + ".npz")


def main():
    loader = DataPreprocess()
    files_num = len(os.listdir(loader.anno_path))
    print("number of files: " + str(files_num))
    # loader.preprocess_nth_file(0)
    # loader.preprocess_nth_file(1)
    for i in range(0, files_num):
        loader.preprocess_nth_file(i)


if __name__ == "__main__":
    main()
