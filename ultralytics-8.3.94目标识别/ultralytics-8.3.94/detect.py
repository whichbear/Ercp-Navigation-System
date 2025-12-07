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

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'E:\yolo\ultralytics-8.3.94\runs\train\exp_ERCPv1\weights\best.pt')
    model.predict(source=r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\images\test',
                  # save_txt设置为 True 时，会将预测结果以文本文件的形式保存下来，文件中会记录每个检测到的目标的类别、边界框坐标等信息。
                  save_txt=True,
                  # conf：置信度阈值，设置为 0.2 表示只有预测置信度大于等于 0.2 的目标才会被保留。
                  conf=0.005,
                  # 交并比（Intersection over Union）阈值，设置为 0.5 用于非极大值抑制（NMS），去除重叠度较高的边界框。
                  iou=0.01,
                  save=True,
                  show=True,
                  )

