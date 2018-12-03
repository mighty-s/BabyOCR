__author__ = 'sight'
import matplotlib.pyplot as plt


def plot_loss(history):
    """
    학습 후 모델의 loss를 표시한 그래프를 출력
    :param history: 
    :return: 
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()


def plot_acc(history):
    """
    학습 후 모델의 정확도를 표시한 그래프를 출력
    :param history:
    :return: 모델의 정확도 그래프 표시
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()
