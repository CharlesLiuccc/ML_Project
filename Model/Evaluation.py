from CNNModel import *
import tensorflow as tf
from tensorflow.python.keras.models import load_model

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_path = "saved/2022-04-26 23'37'01/0/weights.h5"

    print("\nmodel building...")
    model = CNNModel()
    model.build_model()
    model.data_partition(0)
    print("\nmodel loading...")
    model.load_built_model(model_path)
    print("\nmodel testing...")
    model.test()
    model.save_predictions()
    print("\nmodel evaluating...")
    model.evaluate()
    model.save_results_csv()
    print("\nDone.")

