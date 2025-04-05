#### Setup
import tempfile
import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy

from tensorflow import keras

quantize=False

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0


### Float Model
print("\nTraining model in each client ...\n")
client_weights = []
for i in range(2):
    # Define the model architecture.
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    weights = list()
    if quantize:
        # q_aware stands for for quantization aware.
        q_aware_model = tfmot.quantization.keras.quantize_model(model)
        # `quantize_model` requires a recompile.
        q_aware_model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        q_aware_model.fit(
            train_images,
            train_labels,
            epochs=1,
            validation_split=0.1,
        )
        for layer in q_aware_model.layers:
            weights.append(layer.get_weights())
    else:
        model.fit(
            train_images,
            train_labels,
            epochs=1,
            validation_split=0.1,
        )
        for layer in model.layers:
            weights.append(layer.get_weights())

    client_weights.append(weights)

new_weights = list()

print("Calculating average weight ...")
for weights_list_tuple in zip(*client_weights):
    new_weights.append(
        [numpy.array(weights_).mean(axis=0)\
            for weights_ in zip(*weights_list_tuple)])

# print("Average weights : ", new_weights)

# new_weights = numpy.array(new_weights, dtype=object)
# print("new weight shape : ", new_weights.shape)

if quantize:
    init_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])
    # q_aware stands for for quantization aware.
    new_model = tfmot.quantization.keras.quantize_model(model)
    # `quantize_model` requires a recompile.
    new_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
else:
    new_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

i = 0
for layer in new_model.layers:
    layer.set_weights(new_weights[i])
    i+=1
# new_model.set_weights(new_weights)

print("Training new model using average weights ...")

new_model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
new_model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_split=0.1,
)