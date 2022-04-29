# readme

## Environment:

**IDE:**

    Jetbrain Pycharm  2022.1

**Language:**

    python 3.9

**Packages:**

    jams 0.3.4

    keras 2.8.0

    librosa 0.9.1

    tensorflow 2.8.0

    numpy 1.21.5

    pandas 1.4.2

## Data Source:

[GuitarSet | Zenodo](https://zenodo.org/record/3371780#.YmwEddrMKUk)

## Project Structure:

the whole project are divided into 2 parts: Data and Model

**Data:**

    this folder contains the original guitar data and label, class used for preprocess the original data and the preprocessed data can be used in trainning

    GuitarSet: annotation folder records the guitar tab labels in .jams file, audio folder records the audio file of different guitar chords in 3 recording ways

    CQT: this folder stored the data preprocessed by the DataPreprocess.py, all the data are in .npz file type

    DataPreprocess.py:

        this python file implement a class called DataPreprocess, which can transfer the audio data into frequency-image data and then extract the corresponding labels and store the numpy array like data into .npz file

        run the main function to generate all the preprocessed data

**Model:**

    4 python files are contained in this folder: CNNModel.py, DataProcess.py, Metrics.py and Evaluation.py, and 1 folder named saved to store the model

    DataProcess.py:

        implements class for generate the input data for cnn model

    CNNModel.py:

        this python file is used to create the cnn model using the data processed by functions in DataProcess.py

    Metrics.py:

        contains functions used for calculate the precision rate for evaluation

    Evaluation.py:

        to test and evaluate the cnn model trained in CNNModel.py and save the results



## Run the Project

    you can just run the main function in Evaluation.py to test and evaluate the model have been built in saved/2022-04-26 23'37'01

    or you can change the parameters in DataPreprocess.py and run the main function in it to generate your guitar training data. And then change the parameters in CNNModel and run its main function to generate your cnn model and get the evaluation result.


