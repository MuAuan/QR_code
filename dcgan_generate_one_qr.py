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
from PIL import Image
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import cv2
import numpy as np

def edit_contrast(image, gamma):
    """コントラクト調整"""
    look_up_table = [np.uint8(255.0 / (1 + np.exp(-gamma * (i - 128.) / 255.)))
        for i in range(256)]

    result_image = np.array([look_up_table[value]
                             for value in image.flat], dtype=np.uint8)
    result_image = result_image.reshape(image.shape)
    return result_image


def noise_get():
    capture = cv2.VideoCapture(1)
    cv2.namedWindow('frame')
    if capture.isOpened() is False:
        raise("IO Error")
    ret, frame = capture.read()
    while True:
        #ret, frame = capture.read()
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
            #num0=np.float(input[1:4])
            #num1=np.float(input[5:8])
            # コード内容を出力
            print(num0)
            #print(num0,num1,num0*num1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(input),(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            return num0
        if cv2.waitKey(1) >= 0:
            break
        ret, frame = capture.read()

n_colors = 3

def generator_model():
    model = Sequential()

    model.add(Dense(8*8*128, input_shape=(2,))) #1024,100 10
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
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
    #model.add(BatchNormalization())

    return model

def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, n_colors))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image


def combine_images2(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    #image = np.zeros((rows * h,  cols * w, n_colors))
    image = np.zeros((rows * h,  cols * w))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        #image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
        image[i*h:(i+1)*h, j*w:(j+1)*w] = img[:, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def generate(BATCH_SIZE=1):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_22000.h5')
    noise = np.random.uniform(size=[BATCH_SIZE, 2], low=-1.0, high=1.0) ##32*32
    num0=noise_get()
    x=np.float(num0[1:4])
    y=np.float(num0[5:8])
    print(x,y)
    noise[0]=[x,y]  #[-1,0.5]
    generated_images = g.predict(noise)
    plt.imshow(generated_images[0])
    plt.savefig("./gen_images/generator_22000_1.png")
    plt.pause(0.01)
    img_bg = Image.open('./gen_images/generator_22000_1.png')
    #img_bg=generated_images[0]
    
    qr = qrcode.QRCode()
    qr.add_data(str(noise[0]))
    qr.make()
    img = qr.make_image()
    img.save('./gen_images/qrcode_test.png')
    
    qr = qrcode.QRCode(box_size=4)
    qr.add_data(str(noise[0]))
    qr.make()
    img_qr = qr.make_image()

    pos = (img_bg.size[0] - img_qr.size[0], img_bg.size[1] - img_qr.size[1])

    img_bg.paste(img_qr, pos)
    img_bg.save('./gen_images/qr_lena_.png')
    
if __name__ == "__main__":
    generate(BATCH_SIZE=1)
    
