from tensorflow import keras, lite
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import zipfile

from sys import getsizeof
import tempfile

class Net():
    def __init__(self, args):
        self.args = args
        mnist = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

    def loadTFLiteModel(self, tflite_model):
        self.tflite_model = tflite_model
        _, self.tflite_file = tempfile.mkstemp('.tflite')
        with open(self.tflite_file, 'wb') as f:
            f.write(self.tflite_model)
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(self.tflite_file)

        interpreter = lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        all_tensor_details = interpreter.get_tensor_details()
        #interpreter.invoke() >> updated by rini

        return interpreter, all_tensor_details
    
    def train_tflite_model(self, interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        prediction_digits = []
        for i, train_image in enumerate(self.train_images):
            train_image = np.expand_dims(train_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, train_image)
            interpreter.invoke()

            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        prediction_digits = np.array(prediction_digits)
        accuracy = (prediction_digits == self.train_labels).mean()
        return accuracy
    
    def calculateTFLiteTensorAverage(self, client_weights):
        self.current_weight = list()
        for weights_list_tuple in zip(*client_weights):
            self.current_weight.append(
                [np.array(weights_).mean(axis=0)\
                    for weights_ in zip(*weights_list_tuple)])

        return self.current_weight
