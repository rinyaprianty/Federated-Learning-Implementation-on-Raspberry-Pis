from tensorflow import keras, lite
from sys import getsizeof
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import copy

import os
import zipfile
import tempfile

class Net():
    def __init__(self, args):
        self.args = args
        mnist = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

        init_model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28)),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10)
        ])
        #pre-training
        print("\nPre-training model, please wait...\n")

        if args.quantize:
            init_model.compile(optimizer='adam',
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            init_model.fit(
                self.train_images,
                self.train_labels,
                epochs=self.args.local_ep,
                validation_split=self.args.validation_split,
                verbose=False
            )
            # q_aware stands for for quantization aware.
            self.model = tfmot.quantization.keras.quantize_model(init_model)
    
        if args.prune:
            batch_size = 128
            validation_split = 0.1 # 10% of training set will be used for validation set. 
            num_images = self.train_images.shape[0] * (1 - validation_split)
            end_step = np.ceil(num_images / batch_size).astype(np.int32) * args.local_ep
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                        final_sparsity=0.80,
                                                                        begin_step=0,
                                                                        end_step=end_step)
            }

            self.model = tfmot.sparsity.keras.prune_low_magnitude(init_model, **pruning_params)

            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
            ]

            self.model.compile(optimizer='adam',
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            self.model.fit(self.train_images, self.train_labels,
                    batch_size=batch_size, epochs=args.local_ep, validation_split=validation_split,
                    callbacks=callbacks, verbose=False)
            
        else:
            self.model = init_model

        # _, keras_file = tempfile.mkstemp('.h5')
        # keras.models.save_model(self.model, keras_file, include_optimizer=False)
        # print("model size : ", os.path.getsize(keras_file))
        # exit("tes")
        
        self.train()
        self.convertToTFLite()
        if self.args.cobacoba:
            self.cobacoba()

    def convertToTFLite(self):
        if self.args.prune:
            model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
            converter = lite.TFLiteConverter.from_keras_model(model_for_export)
        else:
            converter = lite.TFLiteConverter.from_keras_model(self.model)

        if self.args.quantize:
            converter.optimizations = [lite.Optimize.DEFAULT]
        self.tflite_model = converter.convert()

        _, self.tflite_file = tempfile.mkstemp('.tflite')
        with open(self.tflite_file, 'wb') as f:
            f.write(self.tflite_model)

        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(self.tflite_file)

        # zfile = zipfile.ZipFile(zipped_file)
        # zfile.extractall(os.path.abspath("saves"))

        print("Model Size : ", getsizeof(self.tflite_model))
        print("File Size : ", os.path.getsize(self.tflite_file))
        print("Zipped File Size : ", os.path.getsize(zipped_file))
        # print("Unzipped File Size : ", os.path.getsize("saves".format(zipped_file)))
        
        return None

    def loadTFLiteModel(self, tflite_model):

        interpreter = lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        all_tensor_details = interpreter.get_tensor_details()
        interpreter.invoke()
        return interpreter, all_tensor_details

    def getTFLiteWeight(self, interpreter, all_tensor_details):
        weights = list()
        for tensor_item in all_tensor_details:
            # print(interpreter.tensor(tensor_item["index"])().shape)
            weights.append(interpreter.tensor(tensor_item["index"])())
        return weights

    def train(self):
        self.model.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=self.args.local_ep,
            validation_split=self.args.validation_split,
            verbose=False
        )
        return None

    def test_tflite_model(self, interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Run predictions on every image in the "test" dataset.
        prediction_digits = []
        for i, test_image in enumerate(self.test_images):
            if i % 1000 == 0 and i > 0:
                print('Evaluated on {} results so far.'.format(i))
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        print('\n')
        # Compare prediction results with ground truth labels to calculate accuracy.
        prediction_digits = np.array(prediction_digits)
        accuracy = (prediction_digits == self.test_labels).mean()
        return accuracy

    def Fed_Avg(self, client_weights, tflite_model):
        average_weight = list()
        for i in range(len(client_weights[0])):
            layer_weights = list()
            if len(client_weights) > 1:
                for w in client_weights:
                    layer_weights.append(w[i].astype(np.float64))
                average_weight.append(keras.layers.Average()(layer_weights))
            else:
                average_weight.append(client_weights[0][i])

        ## set average on new model
        interpreter, _ = self.loadTFLiteModel(tflite_model)
        weights = self.getTFLiteWeight(interpreter, _)
        # print("indices : ", indices)
        print("Average weight calculated, setting weight on new model...")
        for i in range(len(weights)):
            try:
                value = np.array(average_weight[i])
                interpreter.set_tensor(i, value)
            except:
                try:
                    value = np.array(average_weight[i]).astype(np.float32)
                    interpreter.set_tensor(i, value)
                except:
                    try:
                        value = np.array(average_weight[i]).astype(np.int32)
                        interpreter.set_tensor(i, value)
                    except:
                        try:
                            value = np.array(average_weight[i]).astype(np.int64)
                            interpreter.set_tensor(i, value)
                        except:
                            try:
                                value = np.array(average_weight[i]).astype(np.int32)
                                interpreter.set_tensor(i, value)
                            except:
                                value = np.array(average_weight[i]).astype(np.int8)
                                interpreter.set_tensor(i, value)

        self.tflite_model = tflite_model
        return average_weight, interpreter





    #################



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

        # Compare prediction results with ground truth labels to calculate accuracy.s
        prediction_digits = np.array(prediction_digits)
        accuracy = (prediction_digits == self.train_labels).mean()
        return accuracy

    def cobacoba(self):
        new_tflite_model = self.tflite_model
        ## training on clients
        client_weights = list()
        for e in range(self.args.epochs):
            for i in range(2):
                client_tflite_model = copy.deepcopy(self.tflite_model)
                interpreter = lite.Interpreter(model_content=client_tflite_model)
                all_tensor_details = interpreter.get_tensor_details()
                interpreter.allocate_tensors()

                accuracy = self.train_tflite_model(interpreter)
                
                print("\nTraining Accuracy : {:.3f}%".format(accuracy * 100))
                print("TFLite model size : {:.3f} MB\n".format(getsizeof(client_tflite_model) / 1000000))
                weights = list()
                for tensor_item in all_tensor_details:
                    weights.append(interpreter.tensor(tensor_item["index"])())
                client_weights.append(weights)

            ## calculate average
            average_weight = list()
            for i in range(len(client_weights[0])):
                layer_weights = list()
                if len(client_weights) > 1:
                    for w in client_weights:
                        layer_weights.append(w[i].astype(np.float64))
                    average_weight.append(keras.layers.Average()(layer_weights))
                else:
                    average_weight.append(client_weights[0][i])
            
            ## set average on new model
            new_interpreter = lite.Interpreter(model_content=new_tflite_model)
            new_interpreter.allocate_tensors()
            input_details = new_interpreter.get_input_details()
            all_tensor_details = new_interpreter.get_tensor_details()
            # interpreter.set_tensor(input_details[0]['index'],
            new_interpreter.invoke()
            for i in range(len(weights)):
                try:
                    value = np.array(average_weight[i])
                    new_interpreter.set_tensor(i, value)
                except:
                    try:
                        value = np.array(average_weight[i]).astype(np.float32)
                        new_interpreter.set_tensor(i, value)
                    except:
                        try:
                            value = np.array(average_weight[i]).astype(np.int32)
                            new_interpreter.set_tensor(i, value)
                        except:
                            try:
                                value = np.array(average_weight[i]).astype(np.int64)
                                new_interpreter.set_tensor(i, value)
                            except:
                                try:
                                    value = np.array(average_weight[i]).astype(np.int32)
                                    new_interpreter.set_tensor(i, value)
                                except:
                                    value = np.array(average_weight[i]).astype(np.int8)
                                    new_interpreter.set_tensor(i, value)
            
            ## evaluate new model
            accuracy = self.test_tflite_model(new_interpreter)
            print("Testing Accuracy : {:.3f}% ".format(accuracy * 100))

        print("\nAccuracy in {} epochs : {:.3f}% ".format(self.args.epochs, (accuracy * 100)))
        exit("nyeh")
        return None