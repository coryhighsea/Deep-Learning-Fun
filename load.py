import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

abalone_train = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
        names=["Length", "Diamter", "Height", "whole weight", "Sucked weight", "Viscera weight", "Shell weight", "Age"])
abalone_train.head()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
abalone_features

abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)

normalize = preprocessing.Normalization()

normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(), 
                            optimizer = tf.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


