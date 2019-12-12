from tensorflow import keras

Input = keras.layers.Input
Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
ZeroPadding2D = keras.layers.ZeroPadding2D
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Flatten = keras.layers.Flatten
BatchNormalization = keras.layers.BatchNormalization
Reshape = keras.layers.Reshape
concatenate = keras.layers.concatenate
Activation = keras.layers.Activation

InputSpec = keras.layers.InputSpec
Layer = keras.layers.Layer
Model = keras.Model

K = keras.backend
preprocess_input = keras.applications.imagenet_utils.preprocess_input
image = keras.preprocessing.image
