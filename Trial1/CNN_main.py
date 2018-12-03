__author__ = 'sight'

import warnings
import sys
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib.pyplot as plt

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

model = load_model('MNIST_CNN_model.h5')
# model.summary()

test1 = plt.imread('C:/Users/sdm32/OneDrive/바탕 화면/플밍/Python/finalTerm/Trial1/runtime_data/KKK.jpg')
# plt.imshow(test1)


test1 = test1.reshape((1, 34, 34, 1))  # 배열을 reshape 하는 함수

print('The Answer is ', model.predict_classes(test1))



'''
test_num = plt.imread('C:/Users/sdm32/OneDrive/바탕 화면/플밍/Python/finalTerm/Trial1/runtime_data/OO.png')
test_num = test_num[:]
test_num = (test_num > 125) * test_num
test_num = test_num.astype('float32') / 255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest')
'''