# -*- coding: utf-8 -*-
"""
@Auth ： HP-Succinum
@File ：detect.py
@IDE ：PyCharm
@Email ：1249140039@qq.com
"""

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    model = YOLO(model=r"C:\Users\19060\Desktop\ultralytics-8.3.94目标识别\ultralytics-8.3.94\ultralytics\cfg\models\11\yolo11.yaml")
    # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    # model.load('yolo11n.pt')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp_ERCPv1',
                single_cls=False,
                cache=False,
                )
