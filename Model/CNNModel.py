from __future__ import print_function

import datetime
import os.path
import random

import keras.optimizers
import numpy as np
import pandas as pd
from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from DataProcess import DataProcess
from Metrics import *


class CNNModel:
    def __init__(self,
                 batch_size=128,
                 epochs=5,
                 con_win_size=9,
                 data_path='../Data/CQT/',
                 id_file="id.csv",
                 save_path=r'saved/'):

        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path

        self.load_IDs()

        # save log information
        self.save_folder = self.save_path + datetime.datetime.now().strftime("%Y-%m-%d %H'%M'%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"

        # model evaluation metrics
        self.metrics = {"pitch_precision": [], "pitch_recall": [], "pitch_f-measure": [],
                        "tab_precision": [], "tab_recall": [], "tab_f-measure": [], "disambiguation_rate": [],
                        "data": ["value"]}

        # input shape
        self.input_shape = (192, self.con_win_size, 1)
        self.num_classes = 21
        self.num_strings = 6

    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def data_partition(self, data_split):
        self.data_split = data_split
        self.partition = {"training": [], "validation": []}

        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)

        # # divide data into training and validation
        # total_files = os.listdir('../Data/CQT')
        # # random.shuffle(total_files)
        # num = len(total_files)
        # carve_num = int(num * 0.8)
        # self.partition["training"] = total_files[:carve_num]
        # self.partition["validation"] = total_files[carve_num:]

        # process input data for model
        self.training_data = DataProcess(self.partition['training'],
                                         data_path=self.data_path,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         con_win_size=self.con_win_size)

        self.validation_data = DataProcess(self.partition['validation'],
                                           data_path=self.data_path,
                                           batch_size=len(self.partition['validation'][0]),
                                           shuffle=True,
                                           con_win_size=self.con_win_size)

        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)

    def log_model(self):
        with open(self.log_file, 'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\nid_file: " + str(self.id_file))
            fh.write("\ncon_win_size: " + str(self.con_win_size) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + "\n"))

    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:, i, :], output[:, i, :])
        return loss

    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        model.compile(loss=self.catcross_by_string,
                      metrics=[self.avg_acc])

        self.model = model

    def load_built_model(self, model_path):
        self.model.load_weights(model_path)

    def train(self):
        self.model.fit_generator(generator=self.training_data,
                                 validation_data=None,
                                 epochs=self.epochs,
                                 verbose=1,
                                 use_multiprocessing=True,
                                 workers=9)

    def save_weights(self):
        self.model.save_weights(self.split_folder + "weights.h5")

    def test(self):
        self.X_test, self.y_gt = self.validation_data[0]
        self.y_pred = self.model.predict(self.X_test)

    def save_predictions(self):
        np.savez(self.split_folder + "prediction.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pitch_precision"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pitch_recall"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pitch_f-measure"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tab_precision"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tab_recall"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tab_f-measure"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["disambiguation_rate"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        output = {}
        print("results: ")
        for key in self.metrics.keys():
            if key != "data":
                output[key] = self.metrics[key]
                print(key + ": " + str(self.metrics[key]))
        output["data"] = self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv")


def main():
    model = CNNModel()

    print("logging model...")
    model.build_model()
    model.log_model()

    guitarist = random.randrange(0, 4)

    print("\nvalidation guitarist number: " + str(guitarist))
    model.data_partition(guitarist)
    print("\nbuilding model...")
    model.build_model()
    print("\ntraining...")
    model.train()
    model.save_weights()
    print("\ntesting...")
    model.test()
    model.save_predictions()
    print("\nevaluation...")
    model.evaluate()

    print("saving results...")
    model.save_results_csv()


if __name__ == "__main__":
    main()
