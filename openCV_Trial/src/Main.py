__author__ = 'sight'

import cv2

img = cv2.imread('../resources/datas.png', cv2.IMREAD_COLOR)
copy_img = img.copy()

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img2, (5, 5), 0)
cv2.imwrite('./manipulated/blur.jpg', blur)

canny = cv2.Canny(blur, 200, 200)
cv2.imwrite('./manipulated/canny.jpg', canny)

ret, binary_img = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY_INV)



'''
Counter 들 찾기..
Counters 는 같은 에너지를 가지는 점들을 연결 한 선
'''
box1 = []
f_count = 0
select = 0
plate_width = 0

cnts, counters, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 전체 이미지
for i in range(len(counters)):
    cnt = counters[i]
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h           # area size
    aspect_ratio = float(w)/h   # ratio = width/height

    if(aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
        cv2.rectangle(binary_img, (x, y), (x+w, y+h), (255, 255, 255), 1)
        box1.append(cv2.boundingRect(cnt))
        crop = img2[y:y+h, x:x+w]
        ret, crop_bi = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow(str(i), crop)
        cv2.imwrite(str(i)+'.png', crop_bi)

    # to find number plate measureing length between rectangles
    for m in range(len(box1)):
        count = 0
        for n in range(m + 1, (len(box1) - 1)):
            delta_x = abs(box1[n + 1][0] - box1[m][0])
            if delta_x > 150:
                break
            delta_y = abs(box1[n + 1][1] - box1[m][1])
            if delta_x == 0:
                delta_x = 1
            if delta_y == 0:
                delta_y = 1
            gradient = float(delta_y) / float(delta_x)
            if gradient < 0.25:
                count = count + 1
            # measure number plate size
            if count > f_count:
                select = m
                f_count = count
                plate_width = delta_x
    # cv2.imwrite('snake.jpg', img)
box1.sort()
cv2.imshow('Original', copy_img)
cv2.imshow('img2', img2)
cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.imshow('snake', img)
cv2.imshow('snake', binary_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
