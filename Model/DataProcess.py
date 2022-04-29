import keras.utils.data_utils
import numpy as np


class DataProcess(keras.utils.data_utils.Sequence):
    def __init__(self,
                 list_IDs,
                 data_path="../Data/CQT/",
                 batch_size=128,
                 shuffle=True,
                 label_dim=(6, 21),
                 con_win_size=9):

        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = (6, 21)
        self.con_win_size = con_win_size
        self.half_win = con_win_size // 2

        self.X_dim = (self.batch_size, 192, self.con_win_size, 1)
        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):

        X = np.empty(self.X_dim)
        y = np.empty(self.y_dim)

        for i, ID in enumerate(list_IDs_temp):

            # load data from .npz file
            data_dir = self.data_path
            file_name = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_index = int(ID.split("_")[-1])

            # load a context window centered around the frame index
            loaded = np.load(data_dir + file_name)
            full_x = np.pad(loaded["audio"], [(self.half_win, self.half_win), (0, 0)], mode='constant')
            sample_x = full_x[frame_index: frame_index + self.con_win_size]
            X[i, ] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)

            # store label
            y[i, ] = loaded["labels"][frame_index]

        return X, y

    def on_epoch_end(self):

        # updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
