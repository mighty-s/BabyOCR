
__author__ = 'sight'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
# custom import
from DataTrain.my_trainer_utils import plot_loss, plot_acc

# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    './dataset/training_set',
    target_size=(34, 34),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale')

valid_datagen = ImageDataGenerator(rescale=1. / 255)

valid_generator = valid_datagen.flow_from_directory(
    './dataset/validation_set',
    target_size=(34, 34),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    './dataset/test_set',
    target_size=(34, 34),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale')

# 2. 모델 구성
model = Sequential()                         # 커널 사이즈 == 필터
model.add(Conv2D(32, kernel_size=(5, 5),     # input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
                 activation='relu',          # (행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
                 input_shape=(34, 34, 1),       
                 padding='same',
                 strides=(1, 1)))
model.add(Conv2D(30, (3, 3), activation='relu'))            # 2차원 컨볼루션 층
model.add(MaxPooling2D(pool_size=(2, 2)))                   # MAxPolling 사용, 2 X 2
model.add(Flatten())                                        # 이미지를 Flat
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(26, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Early Stopping
early = [EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       mode='min')]

# 4. 모델 학습시키기
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=70,
    validation_data=valid_generator,
    validation_steps=5)

# 5. 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# 9. epoch 에 따른 loss, val_loss 그래프
plot_loss(hist)
plot_acc(hist)

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=10)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)

# 7. 만든 모델 저장
model.save('result_model.h5')
