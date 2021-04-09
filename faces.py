import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
        titleweight='bold', titlesize=18, titlepad=10)

#from learntools.core import binder
#binder.bind(globals())
#from learntools.deep_learning_intro.ex2 import *

the_faces = pd.read_csv('concrete.csv')
print(the_faces.head())

input_shape=[8]

model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation(activation='relu'),
    layers.Dense(32),
    layers.Activation(activation='relu'),
    layers.Dense(1)
])

activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x)

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
plt.savefig('output.png')

