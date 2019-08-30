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

if __name__ == "__main__":
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
        if cv2.waitKey(1) >= 0:
            break
     
        ret, frame = capture.read()        