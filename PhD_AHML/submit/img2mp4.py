import glob
import os

import cv2
from PIL import Image


def get_file(root_path, all_files=[], sub_dirpaths=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有目录及文件路径，all_files，all_dirpath
    '''
    filenames = os.listdir(root_path)
    
    for filename in filenames:
        filepath = os.path.join(root_path, filename)
        if not os.path.isdir(filepath):  # not a dir
            all_files.append(filepath)
        else:  # is a dir
            sub_dirpaths.append(filepath)
            get_file(filepath, all_files)
    return all_files, sub_dirpaths


path = r'D:\Desktop\1'
filepaths, dirpathlist = get_file(path)
# print(len(dirpathlist))


for dirpath in dirpathlist[:]:
    fps = 18  # 一秒一帧
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(dirpath)
    filenames = os.listdir(dirpath)
    
    # sort_num_first = []
    # for file in filenames:
    #     # 根据 _ 分割，然后根据空格分割，转化为数字类型
    #     sort_num_first.append(str(file.split(".")[0]))
    # sort_num_first.sort()
    # print(sort_num_first)
    # sorted_file = []
    # for sort_num in sort_num_first:
    #     for file in filenames:
    #         if str(sort_num) == file.split(".")[0]:
    #             sorted_file.append(file)
    # filenames = sorted_file
    
    filepath = os.path.join(dirpath, filenames[0])
    if os.path.isfile(filepath):
        img = Image.open(filepath)
        print(filepath)
        size = img.size  # 大小/尺寸,获取第一张图片的尺寸
        print(size)
        videpath = path + '\\' + 'Demo1.avi'
        videoWriter = cv2.VideoWriter(
            videpath, fourcc, fps, size)  # 视频按照图片尺寸合成
        imgpaths = dirpath + '/*.png'
        imgs = glob.glob(imgpaths)
        for imgname in imgs:
            for i in range(3):  # 一张图循环7次，fps = 1，一张图停留7s
                frame = cv2.imread(imgname)
                videoWriter.write(frame)
        videoWriter.release()
    else:
        continue
