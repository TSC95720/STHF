import torch
from PIL import Image
import os
from torchvision import transforms
from natsort import natsorted
from Filter import fre_aug, save_images


transform_fft = transforms.Compose([
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
])

seq_len = 6

# new_data_path = 'E:/ReID/dataset/VCM_Rec/Train'  # 生成数据数据集保存文件夹
new_data_path = '/dataset/datasets1/HITSZ-VCM_Rec/Train'
for i in range(1, 503, 1):
    no1_dir = str(i).rjust(4, '0')  # 获取文件夹的序号 eg:0001
    # no1_dirpath = 'E:/ReID/dataset/HITSZ-VCM/Train/' + no1_dir  # 原始数据集文件夹
    no1_dirpath = '/dataset/datasets1/HITSZ-VCM/Train/' + no1_dir
    isExists = os.path.exists(no1_dirpath)
    if not isExists:
        continue;
    no2_dir_rgb_path = os.path.join(no1_dirpath, 'rgb')
    no2_dir_ir_path  = os.path.join(no1_dirpath, 'ir')
    no3_dir_rgb = os.listdir(no2_dir_rgb_path)  # 生成此文件夹下的字文件夹名列表（'D1', 'D3', 'D5'）
    for j_r in no3_dir_rgb:  # 依次遍历这些文件夹
        create_path = os.path.join(new_data_path, no1_dir, 'rgb', j_r)  # 新文件夹路径
        isExists = os.path.exists(create_path)
        if not isExists:
            os.makedirs(create_path)  # 创建文件夹
        no3_dir_rgb_path = os.path.join(no2_dir_rgb_path, j_r)  # 原文件夹路径
        imgpath_pick = os.listdir(no3_dir_rgb_path)  # ID文件夹中的下标索引['1.jpg', '101.jpg', '106.jpg',...]
        imgpath_pick = natsorted(imgpath_pick)   # 按时间连续采样

# ---------------------------------High Pass Filter---------------------------------------------------------------------

        # 按序列读取图片
        for frame in range(0, len(imgpath_pick), seq_len):
            images_list = []
            name_list = []
            for k in imgpath_pick[frame:frame + seq_len]:
                readpath = os.path.join(no3_dir_rgb_path, k)  # 图片读取路径
                src_img = Image.open(readpath).convert('L')  # 读取图像
                src_img = transform_fft(src_img)
                if src_img is not None:  # 确保图像读取成功
                    images_list.append(src_img)  # 将图像存储在列表中
                    name_list.append(k)   # 将图像名存储在列表中
            seq = torch.stack(images_list)
            seq = torch.squeeze(seq, dim=1)
            new_seq = fre_aug(seq, 2, 10)
            save_images(new_seq, create_path, name_list)
# make IR data
    no3_dir_ir = os.listdir(no2_dir_ir_path)
    for j_r in no3_dir_ir:
        create_path = os.path.join(new_data_path, no1_dir, 'ir', j_r)  # 文件夹路径
        isExists = os.path.exists(create_path)
        if not isExists:
            os.makedirs(create_path)  # 创建文件夹

        no3_dir_ir_path = os.path.join(no2_dir_ir_path, j_r)
        imgpath_pick = os.listdir(no3_dir_ir_path)
        imgpath_pick = natsorted(imgpath_pick)

# ---------------------------------High Pass Filter---------------------------------------------------------------------

        # 按序列读取图片
        for frame in range(0, len(imgpath_pick), seq_len):
            images_list = []
            name_list = []
            for k in imgpath_pick[frame:frame + seq_len]:
                readpath = os.path.join(no3_dir_ir_path, k)  # 图片读取路径
                src_img = Image.open(readpath).convert('L')  # 读取图像
                src_img = transform_fft(src_img)
                if src_img is not None:  # 确保图像读取成功
                    images_list.append(src_img)  # 将图像存储在列表中
                    name_list.append(k)   # 将图像名存储在列表中
            seq = torch.stack(images_list)
            seq = torch.squeeze(seq, dim=1)
            new_seq = fre_aug(seq, 2, 10)
            save_images(new_seq, create_path, name_list)






