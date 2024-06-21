"code originally in PCN.h"
import numpy as np
import cv2
from enum import IntEnum

FEAT_POINTS = 14

class Window:
    def __init__(self, x, y, width, angle, score, points = []):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score
        self.points = points

class FeatEnum(IntEnum):
    CHIN_0 = 0
    CHIN_1 = 1
    CHIN_2 = 2
    CHIN_3 = 3
    CHIN_4 = 4
    CHIN_5 = 5
    CHIN_6 = 6
    CHIN_7 = 7
    CHIN_8 = 8
    NOSE = 9
    EYE_LEFT = 10
    EYE_RIGHT = 11
    MOUTH_LEFT = 12
    MOUTH_RIGHT = 13
    FEAT_POINTS = 14

def rotate_point(x, y, centerX, centerY, angle):
    # Translate the point to the origin
    x -= centerX
    y -= centerY
    
    # Convert angle from degrees to radians
    theta = -angle * np.pi / 180
    
    # Perform the rotation
    rx = centerX + x * np.cos(theta) - y * np.sin(theta)
    ry = centerY + x * np.sin(theta) + y * np.cos(theta)
    
    return (rx, ry)

CYAN=(255,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREEN=(0,255,0)
YELLOW=(0,255,255)

def draw_line(img, pointlist):
    thick = 2
    int_pointlist = [tuple(map(int, point)) for point in pointlist]
    cv2.line(img, int_pointlist[0], int_pointlist[1], CYAN, thick)
    cv2.line(img, int_pointlist[1], int_pointlist[2], CYAN, thick)
    cv2.line(img, int_pointlist[2], int_pointlist[3], CYAN, thick)
    cv2.line(img, int_pointlist[3], int_pointlist[0], BLUE, thick)

def draw_face(img, face:Window):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x -1
    y2 = face.width + face.y -1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    draw_line(img, pointlist)

def draw_points(img, face:Window):
    thick = 2
    f = FeatEnum.NOSE
    cv2.circle(img,(int(face.points[f][0]),int(face.points[f][1])),thick,GREEN,-1)
    f = FeatEnum.EYE_LEFT
    cv2.circle(img,(int(face.points[f][0]),int(face.points[f][1])),thick,YELLOW,-1)
    f = FeatEnum.EYE_RIGHT
    cv2.circle(img,(int(face.points[f][0]),int(face.points[f][1])),thick,YELLOW,-1)
    f = FeatEnum.MOUTH_LEFT
    cv2.circle(img,(int(face.points[f][0]),int(face.points[f][1])),thick,RED,-1)
    f = FeatEnum.MOUTH_RIGHT
    cv2.circle(img,(int(face.points[f][0]),int(face.points[f][1])),thick,RED,-1)
    for i in range(8):
        cv2.circle(img,(int(face.points[i][0]),int(face.points[i][1])),thick,BLUE,-1)

'''
def draw_points2(img, face:Window):
    thick = 2;
    cyan = (255, 255, 0)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    purple = (255, 0, 139)
    red = (0, 0, 255)
    if len(face.points) == 14:
        for i in range(1, 9):
            cv2.line(img, face.points[i - 1], face.points[i], blue, thick)

        for i in range(len(face.points)):
            if i <= 8:
                cv2.circle(img, face.points[i], thick, cyan, -1)
            elif i <= 9:
                cv2.circle(img, face.points[i], thick, green, -1)
            elif i <= 11:
                cv2.circle(img, face.points[i], thick, purple, -1)
            else:
                cv2.circle(img, face.points[i], thick, red, -1)
'''

def crop_face(img, face:Window, crop_size=200):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x - 1
    y2 = face.width + face.y - 1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    srcTriangle = np.array([
        pointlist[0],
        pointlist[1],
        pointlist[2],
    ], dtype=np.float32)
    dstTriangle = np.array([
        (0, 0),
        (0, crop_size - 1),
        (crop_size - 1, crop_size - 1),
    ], dtype=np.float32)
    rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle)
    ret = cv2.warpAffine(img, rotMat, (crop_size, crop_size))
    return ret, pointlist