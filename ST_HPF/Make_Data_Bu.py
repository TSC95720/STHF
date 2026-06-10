import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from Filter import fre_aug, save_images

# 原始数据集路径
src_root = '/data1/ls/data/BUPTCampus/DATA/'
# 目标数据集路径
dst_root = '/data1/ls/data/BUPTCampus/DATA_Rec/'

# 处理参数
seq_len = 6

# 图像变换（FFT 预处理）
transform_fft = transforms.Compose([
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
])


def process_rgb_images():
    """
    遍历数据集，处理RGB文件夹下的所有摄像头文件夹图片，并存储到目标路径
    """
    # 遍历 1-3080 号文件夹
    for folder_id in range(1, 3081):
        src_folder_path = os.path.join(src_root, str(folder_id))
        dst_folder_path = os.path.join(dst_root, str(folder_id))

        if not os.path.exists(src_folder_path):
            continue

        # 仅处理 RGB 文件夹
        src_rgb_path = os.path.join(src_folder_path, "RGB")
        dst_rgb_path = os.path.join(dst_folder_path, "RGB")

        if not os.path.exists(src_rgb_path):
            continue

        os.makedirs(dst_rgb_path, exist_ok=True)  # 创建目标 RGB 目录

        # 遍历所有摄像头文件夹
        for camera_folder in os.listdir(src_rgb_path):
            src_camera_path = os.path.join(src_rgb_path, camera_folder)
            dst_camera_path = os.path.join(dst_rgb_path, camera_folder)

            if not os.path.isdir(src_camera_path):
                continue

            os.makedirs(dst_camera_path, exist_ok=True)  # 创建目标摄像头目录

            # 获取该摄像头文件夹下所有图片
            img_list = sorted(os.listdir(src_camera_path))

            # 按序列读取图片
            for frame in range(0, len(img_list), seq_len):
                images_list = []
                name_list = []
                for img_name in img_list[frame:frame + seq_len]:
                    readpath = os.path.join(src_camera_path, img_name)
                    src_img = Image.open(readpath).convert('L')  # 读取并转换为灰度
                    src_img = transform_fft(src_img)  # 进行 FFT 变换
                    if src_img is not None:
                        images_list.append(src_img)
                        name_list.append(img_name)

                if images_list:
                    seq = torch.stack(images_list)  # 组合成张量
                    seq = torch.squeeze(seq, dim=1)
                    new_seq = fre_aug(seq, 2, 10)  # 进行频率增强
                    save_images(new_seq, dst_camera_path, name_list)  # 保存处理后的图像


def process_ir_images():
    """
    遍历数据集，处理RGB文件夹下的所有摄像头文件夹图片，并存储到目标路径
    """
    # 遍历 1-3080 号文件夹
    for folder_id in range(1, 3081):
        src_folder_path = os.path.join(src_root, str(folder_id))
        dst_folder_path = os.path.join(dst_root, str(folder_id))

        if not os.path.exists(src_folder_path):
            continue

        # 仅处理 RGB 文件夹
        src_rgb_path = os.path.join(src_folder_path, "IR")
        dst_rgb_path = os.path.join(dst_folder_path, "IR")

        if not os.path.exists(src_rgb_path):
            continue

        os.makedirs(dst_rgb_path, exist_ok=True)  # 创建目标 RGB 目录

        # 遍历所有摄像头文件夹
        for camera_folder in os.listdir(src_rgb_path):
            src_camera_path = os.path.join(src_rgb_path, camera_folder)
            dst_camera_path = os.path.join(dst_rgb_path, camera_folder)

            if not os.path.isdir(src_camera_path):
                continue

            os.makedirs(dst_camera_path, exist_ok=True)  # 创建目标摄像头目录

            # 获取该摄像头文件夹下所有图片
            img_list = sorted(os.listdir(src_camera_path))

            # 按序列读取图片
            for frame in range(0, len(img_list), seq_len):
                images_list = []
                name_list = []
                for img_name in img_list[frame:frame + seq_len]:
                    readpath = os.path.join(src_camera_path, img_name)
                    src_img = Image.open(readpath).convert('L')  # 读取并转换为灰度
                    src_img = transform_fft(src_img)  # 进行 FFT 变换
                    if src_img is not None:
                        images_list.append(src_img)
                        name_list.append(img_name)

                if images_list:
                    seq = torch.stack(images_list)  # 组合成张量
                    seq = torch.squeeze(seq, dim=1)
                    new_seq = fre_aug(seq, 2, 10)  # 进行频率增强
                    save_images(new_seq, dst_camera_path, name_list)  # 保存处理后的图像


# 运行数据处理
process_rgb_images()
process_ir_images()

print('Complete !!!')
