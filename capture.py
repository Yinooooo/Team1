from time import time
import cv2

import os
import numpy as np
import torch
from PIL import Image
import time

from models.experimental import attempt_load
import my_yolov5_detect as detect


cap = cv2.VideoCapture(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cpu')

print("load models ...\n")

milieu_weights = 'C:\\Users\\Ken\\Downloads\\results\\trained_models\\baby_face.pt'

milieu_model_yolo = attempt_load(milieu_weights, map_location=device)

milieu_names = milieu_model_yolo.names

rnd = np.random.RandomState(123)
milieu_colors = [[rnd.randint(0, 255) for _ in range(3)]
                 for _ in range(len(milieu_names))]


def information_collect(obj_info):  # 回傳 內容與等級
    # 準備要回傳到前端的資訊字典格式
    obj_list = []
    for obj in obj_info:
        # print(obj["obj_name"])
        obj_list.append(obj['obj_name'])

    print(obj_list)

    # 回傳內容
    response = ''

    # 回傳危險資料與等級
    level = ''

    if obj_info == None:
        if (len(obj_info)) == 0:
            return response, ''

    if obj_list.count('Human_face') != 1:
        response += '臉 '
        if level == '':
            level = '極度危險'
    if obj_list.count('Human_mouth') != 1:
        response += '嘴巴 '
        if level == '':
            level = '紅燈'
    if obj_list.count('Human_nose') != 1:
        response += '鼻子 '
        if level == '':
            level = '紅燈'
    if obj_list.count('Human_eye')+obj_list.count('Close_eye') == 0:
        response += '眼睛 '
        if level == '':
            level = '黃燈'

    # 假如都沒有東西
    if response == '':
        response = '安全'
    if level == '':
        level = '綠燈'
    return response, level


while(True):
    ref, frame = cap.read()

    # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_result, obj_info = detect.yolo_detect_coco(
        frame, milieu_model_yolo)  # 呼叫人臉偵測

    response, level = information_collect(obj_info)

    #print(response, level)

    cv2.imshow('frame', img_result)

    time.sleep(0.033)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
