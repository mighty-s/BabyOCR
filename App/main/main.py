__author__ = 'sight'
import cv2
import numpy as np
from keras.models import load_model
import App.main.myHeader as my

word_map = {0:  'A', 1:  'B', 2:  'C',  3: 'D',  4: 'E',  5: 'F',  6: 'G',
            7:  'H', 8:  'I', 9:  'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
            22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = load_model('./MNIST_CNN_model.h5')
model.summary()

img = cv2.imread('../resources/CBH.jpg', cv2.IMREAD_COLOR)
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 200, 200)

ret, thresh = cv2.threshold(blur, 120, 255, 0)
ret_bi, binary_img = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY_INV)

# contour 찾기
cnts, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 잘린 이미지들이 있는 배열
boxes = []
container = []          # 사각형처진 글자들이 담겨있는 컨테이너
answerList = []         # 정답 목록

# contour 리스틑 을 순회 하며 contour 주위를 사각형 친다.
# 해당 box의 좌표를 boxes 에 기록
my.search_contours(contours, boxes, img)

# 박스 리스트에 있는거 정렬 - x좌표 기준
boxes = my.letter_sort(boxes)

print('함수 탈출 후 컨테이너 개수 :', len(boxes))

# 박스 리스트에 있는 순서대로 이미지 잘라서 집어 넣기
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    crop = binary_img[y - 5:y + h + 5, x + 1:x + w + 1]
    container.append(crop)

for i in range(len(container)):
    target = cv2.resize(container[i], dsize=(34, 34), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(str(i), target)
    target = target.reshape((1, 34, 34, 1))
    answer = model.predict_classes(target)
#    print('The Answer is ', i, ':', answer)
    answerList.append(answer[0])


print(answerList)
for i in range(len(answerList)):
    print(word_map[answerList[i]], end='')

cv2.imshow('original', img)
cv2.imshow('gray', gray)
cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.imshow('binary black', binary_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
