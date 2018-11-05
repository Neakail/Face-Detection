# -*- coding:utf-8
import cv2
import sys
from PIL import Image


def opencvapi(framename):
    frame = cv2.imread(framename)


    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    #将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(faceRects) > 0:            #大于0则检测到人脸
        for faceRect in faceRects:  #单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

    cv2.imwrite('2/'+framename,frame)
