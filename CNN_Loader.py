__author__ = 'sight'

import warnings
import sys
import tensorflow as tf
import keras
from CNN_Trainer import CNN

warnings.simplefilter(action='ignore', category=FutureWarning)
print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

from keras.models import load_model

model = load_model('MNIST_CNN_model.h5')  # CNN Trainer 에서 훈련시킨 Model 을 가져온다.
model.summary()




