__author__ = 'sight'
import cv2
import itertools
import numpy


def search_contours(contours, container, image):
    """

    :param contours:
    :param container:
    :param image:
    :return:
    """
    print('카운터 개수 :', len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)  # 원본에 파랑 네모침 (웬진 모르지만 BGR) 임
        container.append((x, y, w, h))

    print('컨테이너 개수 :', len(container))


def letter_sort(boxes):
    """

    :param boxes:   박스 (x, y, w, h)를 엘리먼트로 가지고 있는 리스트
    :return:        (영어,한글 읽는 순서)로 정렬된 문자 배열
    """
    word = []
    boxes = sorted(boxes, key=lambda box: box[1])  # 박스들을 Y좌표 기준으로 정렬..
    #  ㄴ> TODO 리스트가 reassign 됨.. 개선 필요 근데 귀찮

    print('정렬 후 컨테이너 개수 :', len(boxes))
    pre_i = 0
    for i in range(len(boxes)):

        if (i != 0) and boxes[i][1]-boxes[i-1][1] >= 30:   # 이전 값이랑 Y 좌표가 40 픽셀 차이나면
            line = boxes[pre_i: i-1]
            line.sort(key=lambda char: char[0])
            print(line)
            word.append(line)
            pre_i = i

        if i == len(boxes)-1:     # 마지막 줄 처리
            line = boxes[pre_i: i - 1]
            pre_i = i
            line.sort(key=lambda char: char[0])
            print(line)
            word.append(line)

    return list(itertools.chain.from_iterable(word))         # 2차원 배열 flat 시켜서 리턴






'''
def letter_sort(boxes):
    """

    :param boxes:   박스 (x, y, w, h)를 엘리먼트로 가지고 있는 리스트
    :return:        (영어,한글 읽는 순서)로 정렬된 문자 배열
    """
    word = []
    boxes = sorted(boxes, key=lambda box: box[1])  # 박스들을 Y좌표 기준으로 정렬..
    #  ㄴ> TODO 리스트가 reassign 됨.. 개선 필요 근데 귀찮
    print(boxes)
    line_count = 0
    for i in range(len(boxes)):
        if (i != 0) and boxes[i][1]-boxes[i-1][1] >= 40:   # 이전 값이랑 Y 좌표가 40 픽셀 차이나면
            print(boxes[i])
            print(boxes[i][1], ' : ', boxes[i-1][1] + 10)
            line_count += 1                                  # 한줄 띄운다
        word[line_count].append(boxes[i])                    # 현재 줄 위치에 박스 삽입
        print(word)

    return list(itertools.chain.from_iterable(word))         # 2차원 배열 flat 시켜서 리턴
'''
