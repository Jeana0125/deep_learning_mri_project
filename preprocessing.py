import random

import numpy as np
from PIL import Image
import os,shutil
import cv2
import csv


def calculate(image1, image2):
    """
    두 이미지 비교
    :param image1: black_image
    :param image2: file_image
    :return: 이미지 비교 점수
    """
    image1 = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(np.asarray(image2), cv2.COLOR_RGB2BGR)

    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 유사도 측정
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_black_image(image1, image2):
    """
    블랙 이미지 추출
    :param image1: black_image
    :param image2: file_image
    :return: 이미지 비교 점수
    """
    h, w, c = image2.shape

    image2 = cv2.cvtColor(np.asarray(image2), cv2.COLOR_RGB2BGR)
    image1 = cv2.resize(image1, (h, w))

    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0

    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def read_csv():
    label_list = {}
    # CSV file 읽어오기
    f = open('mir_csv/train_labels.csv', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        label_list[line[0]] = line[1]
    f.close()
    return label_list


def classify_save_image(patient_num,file_type,im,image,label_list):
    """
     이미지 저장
    :param patient_num: 폴더명 (예:00000,00003...)
    :param file_type: 폴더명(예:FLAIR,T1w...)
    :param im: image name
    :param image: 저장하려는 이미지
    :param label_list: train_labels.csv 읽어온 label list
    :return: none
    """
    file_name = patient_num + '_' + file_type + '_' + im
    print(file_name)

    label = label_list[patient_num]
    print(label)

    if label == '0':
        cv2.imwrite(r"E:\mri_pre_data\t\\" + file_name, image)
    elif label == '1':
        cv2.imwrite(r"E:\mri_pre_data\f\\" + file_name, image)


def pretraining_data():
    # black image
    data = np.zeros([32, 32, 3], dtype=np.uint8)
    black_image = cv2.rectangle(data, (0, 0), (32, 32), (0, 0, 0), 3)
    image1 = cv2.cvtColor(np.asarray(black_image), cv2.COLOR_RGB2BGR)

    # 이미지 읽어오기
    path = 'mri_file/train_jpg0503/'
    files = os.listdir(path)

    #train_labels 읽어오기
    label_list = read_csv()

    for file in files:
        file_types = os.listdir(path + file)
        for file_type in file_types:
            file_addr = path + file + '/' + file_type + '/'
            print(file_addr)
            image_list = os.listdir(file_addr)
            for im in image_list:
                image = cv2.imread(file_addr + im)
                result = classify_black_image(image1, image)
                if result < 1:
                    print(result)
                    classify_save_image(file, file_type, im, image, label_list)


def pre_fine_tuning_data():
    """
    threshold = 0.00777213(fine_tuning_data), 0.00774576(test_data) 기준으로 이미지 선택
    :return:
    """
    # black image
    data = np.zeros([32, 32, 3], dtype=np.uint8)
    black_image = cv2.rectangle(data, (0, 0), (32, 32), (0, 0, 0), 3)
    image1 = cv2.cvtColor(np.asarray(black_image), cv2.COLOR_RGB2BGR)

    # 이미지 읽어오기
    #path = 'mri_file/train_jpg0503/'
    path = 'E:\\test_jpg\\'
    files = os.listdir(path)
    for file in files:
        file_types = os.listdir(path + file)

        for file_type in file_types:
            file_addr = path + file + '/' + file_type + '/'
            print(file_addr)
            image_list = os.listdir(file_addr)
            for im in image_list:
                image = cv2.imread(file_addr + im)
                result = classify_black_image(image1, image)
                if result < 0.00777213:
                    print(result)
                    if not os.path.exists("E:\pre_test_data\\" + file):
                        print(file + 'not exits')
                        os.makedirs("E:\pre_test_data\\" + file)
                    if not os.path.exists("E:\pre_test_data\\" + file + "\\" + file_type):
                        print(file + file_type + 'exits')
                        os.makedirs("E:\pre_test_data\\" + file + "\\" + file_type)

                    cv2.imwrite(r"E:\pre_test_data\\" + file + "\\" + file_type + "\\" + im, image)



def fine_tuning_data():
    """
    매 환자별, 매 타입별 64개 이미지 랜덤 추출(64개 부족시 모든 이미지 포함)
    """
    # train_labels 읽어오기
    label_list = read_csv()

    path = "E:\pre_fine_tuning_data\\"
    files = os.listdir(path)
    for file in files:
        file_types = os.listdir(path + file)
        for file_type in file_types:
            file_addr = path + file + '/' + file_type + '/'
            print(file_addr)
            image_list = os.listdir(file_addr)
            pick_number = 64
            image_list_size = len(image_list)
            print(image_list_size)
            if image_list_size > pick_number:
                sample = random.sample(image_list,pick_number)
                print(sample)
                for name in sample:
                    file_name = file + '_' + file_type + '_' + name
                    print(file_name)

                    label = label_list[file]
                    print(label)

                    if label == '0':
                        shutil.copy(file_addr+name,"E:\\tuning_data\\f\\"+file_name)
                    elif label == '1':
                        shutil.copy(file_addr+name,"E:\\tuning_data\\t\\"+file_name)
            else:
                raw_image_list = os.listdir(file_addr)
                for image in raw_image_list:
                    file_name = file + '_' + file_type + '_' + image
                    print(file_name)

                    label = label_list[file]
                    print(label)

                    if label == '0':
                        shutil.copy(file_addr + image, "E:\\tuning_data\\f\\" + file_name)
                    elif label == '1':
                        shutil.copy(file_addr + image, "E:\\tuning_data\\t\\" + file_name)


def mri_test_data():

    path = "E:\pre_test_data\\"
    files = os.listdir(path)
    for file in files:
        file_types = os.listdir(path + file)
        if not os.path.exists("E:\\test_data\\" + file):
            print(file + 'not exits')
            os.makedirs("E:\\test_data\\" + file)

        for file_type in file_types:
            file_addr = path + file + '/' + file_type + '/'
            print(file_addr)
            image_list = os.listdir(file_addr)
            pick_number = 10
            image_list_size = len(image_list)
            print(image_list_size)
            if image_list_size > pick_number:
                sample = random.sample(image_list, pick_number)
                print(sample)
                for name in sample:
                    file_name = file + '_' + file_type + '_' + name
                    print(file_name)
                    shutil.copy(file_addr + name, "E:\\test_data\\"+file +"\\"+ file_name)
            else:
                raw_image_list = os.listdir(file_addr)
                for image in raw_image_list:
                    file_name = file + '_' + file_type + '_' + image
                    print(file_name)
                    shutil.copy(file_addr + image, "E:\\test_data\\"+file +"\\"+ file_name)


if __name__ == '__main__':
    #pretraining_data()
    #pre_fine_tuning_data()
    #fine_tuning_data()
    mri_test_data()





