import cv2 as cv
import re
import os
import math
import numpy as np
import pandas as pd

from sklearn import svm

# 灰度共生矩阵的计算

# 定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print(height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape
    max_gray_level = maxGrayLevel(input)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)
    return ret

def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm

# 返回 4个变量
def testfeature(img):
    # img = cv.imread(image_name)
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return
    img = cv.resize(img, (250, 250), interpolation=cv.INTER_AREA)  # 500, 400 均划为300×300格式, 双线性插值法
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)
    asm, con, eng, idm = feature_computer(glcm_0)
    if (not asm) or (not con)  or (not eng) or (not idm):
        print('没有返回值！')

    return asm, con, eng, idm  # 角二阶矩（能量）、对比度、熵、反差分矩阵（逆方差）

# 返回R,G,B、H、L、S: N×6的矩阵
def reRGBandHLS(img):
    im_B = img[:, :, 0]  # 从RGB中读入通道
    im_G = img[:, :, 1]
    im_R = img[:, :, 2]
    # R, G, B
    im_R_mean = np.mean(im_R)
    im_G_mean = np.mean(im_G)
    im_B_mean = np.mean(im_B)
    # 转化为HLS颜色空间
    img_hls=cv.cvtColor(img, cv.COLOR_BGR2HLS)
    im_H = img[:, :, 0]  # 从HLS中读入通道
    im_L = img[:, :, 1]
    im_S = img[:, :, 2]
    # H, L, S
    im_H_mean = np.mean(im_H)
    im_L_mean = np.mean(im_L)
    im_S_mean = np.mean(im_S)
    return im_R_mean,im_G_mean,im_B_mean,im_H_mean,im_L_mean,im_S_mean
    pass

if __name__ == '__main__':

    ims_path='corn_new_0/'# 图像数据集的路径
    ims_list=os.listdir(ims_path)
    # print(ims_list)
    imgnum=len(ims_list)  # 获取总个数
    data=np.zeros((imgnum, 11))  # 角二阶矩（能量）、对比度、熵、反差分矩阵（逆方差）、R,G,B、H、L、S、 TAG（0机械损伤，1霉变，2虫蛀）
    for i in range(imgnum):
        im=ims_list[i]
        # 添加标签
        if r'hurt' in im:
            data[i, 10]=0
        elif r'mildew' in im:
            data[i, 10] = 1
        elif r'worm' in im:
            data[i, 10] = 2
        img = cv.imread('corn_new_0/'+im)
        # print('img', img)
        # 0,1,2,3 添加灰度矩阵四类特征
        asm, con, eng, idm = testfeature(img)
        data[i, 0:4]=asm, con, eng, idm
        # 4-9 添加6类图像空间特征
        data[i, 4:10]=reRGBandHLS(img)

    print('All data', data)
    print(type(data))  # 显示此变量类型
    print('data size', data.shape)  # 输出此data纬度
    data_tosave=pd.DataFrame(data)
    data_tosave.to_csv('data/data0.csv')
    print('success saving!')


















    # path1='corn_new_0/hurt1.png'
    # rgb1=cv.imread(path1)
    # print('rgb\n:', rgb1)
    # # cv.imshow("rgb", rgb1)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    # hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
    # print('hsv\n:', hsv1)
    # # cv.imshow("hsv", hsv1)
    # # cv.waitKey(0)
    # hls1=cv.cvtColor(rgb1, cv.COLOR_BGR2HLS)
    # print('hls1\n:', hls1)
    # cv.imshow("hls", hls1)
    # cv.waitKey(0)
'''
    ims_path='corn_new_0/'# 图像数据集的路径
    ims_list=os.listdir(ims_path)
    R_means=[]
    G_means=[]
    B_means=[]
    for im_list in ims_list:
        im=cv.imread(ims_path+im_list)
    #extrect value of diffient channel
        im_B=im[:,:,0]      # 从RGB中读入通道
        im_G=im[:,:,1]
        im_R=im[:,:,2]
    #count mean for every channel
        im_R_mean=np.mean(im_R)
        im_G_mean=np.mean(im_G)
        im_B_mean=np.mean(im_B)
    #save single mean value to a set of means
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        print('图片：{} 的 RGB平均值为 \n[{}，{}，{}]'.format(im_list,im_R_mean,im_G_mean,im_B_mean) )
    #three sets  into a large set
    a=[R_means,G_means,B_means]
    mean=[0,0,0]
    #count the sum of different channel means
    mean[0]=np.mean(a[0])
    mean[1]=np.mean(a[1])
    mean[2]=np.mean(a[2])
    print('数据集的BGR平均值为\n[{}，{}，{}]'.format( mean[0], mean[1], mean[2]))
    # a1, a2, a3, a4 = testfeature("corn_new_0/hurt1.png")
    # print(a1, a2, a3, a4)
    # # asm


    asm_train=[]
    con_train=[]
    eng_train=[]
    idm_train=[]
    ans_train=[]

    asm_test = []
    con_test = []
    eng_test = []
    idm_test = []
    ans_test = []

    # 机械损伤1-10 train, 11-14 test, 用0表示
    for i in range(1, 11):
       # path1='corn_new_0/hurt'+str(i)+'.png'
        path_new = 'corn_new_0/' + 'hurt' + str(i) + '.png'
        asmt, cont, engt, idmt = testfeature(path_new)
        asm_train.append(asmt)
        con_train.append(cont)
        eng_train.append(engt)
        idm_train.append(idmt)
        ans_train.append(0)
        print('hurt', i, 'finished!')

    for i in range(11, 15):
        path1 = r'corn_new_0/hurt'+str(i)+'.png'
        asm, con, eng, idm = testfeature(path1)
        asm_test.append(asm)
        con_test.append(con)
        eng_test.append(eng)
        idm_test.append(idm)
        ans_test.append(0)
        print('hurt', i, 'finished!')

    # 霉变，用1表示, 1-10 trian, 11-15 test
    for i in range(1, 11):
        path1 = r'corn_new_0/mildew' + str(i) + '.png'
        asm, con, eng, idm = testfeature(path1)
        asm_train.append(asm)
        con_train.append(con)
        eng_train.append(eng)
        idm_train.append(idm)
        ans_train.append(1)
        print('mildew', i, 'finished!')

    for i in range(11, 16):
        path1 = r'corn_new_0/mildew' + str(i) + '.png'
        asm, con, eng, idm = testfeature(path1)
        asm_test.append(asm)
        con_test.append(con)
        eng_test.append(eng)
        idm_test.append(idm)
        ans_test.append(1)
        print('mildew', i, 'finished!')

    # 虫蛀，用2表示，1-10 train, 11-20 test
    for i in range(1, 11):
        path1 = r'corn_new_0/worm' + str(i) + '.png'
        asm, con, eng, idm = testfeature(path1)
        asm_train.append(asm)
        con_train.append(con)
        eng_train.append(eng)
        idm_train.append(idm)
        ans_train.append(2)
        print('worm', i, 'finished!')

    for i in range(11, 21):
        path1 = r'corn_new_0/worm' + str(i) + '.png'
        asm, con, eng, idm = testfeature(path1)
        asm_test.append(asm)
        con_test.append(con)
        eng_test.append(eng)
        idm_test.append(idm)
        ans_test.append(2)
        print('worm', i, 'finished!')

    # 输出数据
    print('train:')
    print('asm:', asm_train)
    print('con:', con_train)
    print('eng:', eng_train)
    print('idm:', idm_train)
    print('ans:', ans_train)

    print('test:')
    print('asm:', asm_test)
    print('con:', con_test)
    print('eng:', eng_test)
    print('idm:', idm_test)
    print('ans:', ans_test)

    #
'''


