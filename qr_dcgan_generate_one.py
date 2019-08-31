from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import Adam,SGD
import numpy as np
from PIL import Image
import os
import glob
import random
import argparse
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Dropout
from keras.models import Model
import qrcode
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

def edit_contrast(image, gamma):
    """コントラクト調整"""
    look_up_table = [np.uint8(255.0 / (1 + np.exp(-gamma * (i - 128.) / 255.)))
        for i in range(256)]

    result_image = np.array([look_up_table[value]
                             for value in image.flat], dtype=np.uint8)
    result_image = result_image.reshape(image.shape)
    return result_image

def noise_get(ret,frame):
    while True:
        if ret == False:
            continue
        # グレースケール化してコントラクトを調整する
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = edit_contrast(gray_scale, 5)

        # 加工した画像からフレームQRコードを取得してデコードする
        codes = decode(image)
        if len(codes) > 0:
            input=codes[0][0].decode('utf-8', 'ignore')
            num0=input[1:len(input)-1]
            print(num0)
            return num0,frame
        if cv2.waitKey(1) >= 0:
            break
        ret, frame = capture.read()

n_colors = 3

def generator_model():
    model = Sequential()

    model.add(Dense(8*8*128, input_shape=(2,))) #1024,100 10
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(n_colors,(5, 5), activation='tanh', strides=2, padding='same'))
    return model

def generate(BATCH_SIZE=1,num0=[1,0]):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_22000.h5')
    noise = np.random.uniform(size=[BATCH_SIZE, 2], low=-1.0, high=1.0) ##32*32
    x=np.float(num0[0:3])
    y=np.float(num0[5:9])
    print(x,y)
    noise[0]=[x,y]  #[-1,0.5]
    generated_images = g.predict(noise)
    return generated_images

if __name__ == "__main__":
    while True:
        capture = cv2.VideoCapture(1)
        cv2.namedWindow('frame')
        if capture.isOpened() is False:
            raise("IO Error")
        ret, frame = capture.read()
        num0,frame=noise_get(ret,frame)
        generated_images=[]
        generated_images=generate(BATCH_SIZE=1,num0=num0)
        plt.imshow(frame)
        plt.axis("off")
        plt.savefig('./gen_images/frame.png')
        plt.close()
        frame = Image.open('./gen_images/frame.png')
        plt.imshow(generated_images[0])
        plt.axis("off")
        plt.savefig('./gen_images/generated_images.png')
        plt.close()
        img = Image.open('./gen_images/generated_images.png').resize(frame.size)
        mask = Image.new("L", frame.size, 1)
        im = Image.composite(frame, img, mask)
        im.save('./gen_images/qr_'+str(num0)+'.png')
        imgArray = np.asarray(im)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgArray,str(num0),(100,200), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',imgArray)
        if cv2.waitKey(1) >= 0:
            break
    
        
    
    
