import os
import shutil
from sklearn.model_selection import train_test_split
import labelimg
# 设置参数
val_size = 0.2
test_size = 0.2
postfix = 'jpg'
imgpath = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\images'
txtpath = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\labels'

output_train_img_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\images/train'
output_val_img_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\images/val'
output_test_img_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\images/test'
output_train_txt_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\labels/train'
output_val_txt_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\labels/val'
output_test_txt_folder = r'E:\yolo\ultralytics-8.3.94\dataset\ERCP_Dataset\labels/test'
w# 创建输出文件夹
os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_test_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)
os.makedirs(output_test_txt_folder, exist_ok=True)

# 获取标签文件列表
listdir = [i for i in os.listdir(txtpath) if 'txt' in i]

# 划分数据集
train_val, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=0)
train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), shuffle=True, random_state=0)


# 定义文件复制函数
def copy_files(file_list, img_src_folder, txt_src_folder, img_dst_folder, txt_dst_folder):
    for i in file_list:
        try:
            img_source_path = os.path.join(img_src_folder, '{}.{}'.format(i[:-4], postfix))
            txt_source_path = os.path.join(txt_src_folder, i)
            img_destination_path = os.path.join(img_dst_folder, '{}.{}'.format(i[:-4], postfix))
            txt_destination_path = os.path.join(txt_dst_folder, i)
            shutil.copy(img_source_path, img_destination_path)
            shutil.copy(txt_source_path, txt_destination_path)
        except FileNotFoundError:
            print(f"文件 {img_source_path} 或 {txt_source_path} 未找到。")
        except Exception as e:
            print(f"复制文件时发生错误: {e}")


# 复制训练集文件
copy_files(train, imgpath, txtpath, output_train_img_folder, output_train_txt_folder)

# 复制验证集文件
copy_files(val, imgpath, txtpath, output_val_img_folder, output_val_txt_folder)

# 复制测试集文件
copy_files(test, imgpath, txtpath, output_test_img_folder, output_test_txt_folder)
print("数据集划分完成！")
