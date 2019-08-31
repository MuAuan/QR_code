from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def edit_contrast(image, gamma):
    """コントラスト調整"""
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
        b,g,r,a = 0,255,0,0 #B(青)・G(緑)・R(赤)・A(透明度)
        ## Use HGS創英角ゴシックポップ体標準 to write Japanese.
        fontpath ='C:\Windows\Fonts\HGRPP1.TTC' # Windows10 だと C:\Windows\Fonts\ 以下にフォントがあります。
        font = ImageFont.truetype(fontpath, 16) # フォントサイズが32
        
        if ret == False:
            continue
            
        # グレースケール化してコントラストを調整する
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = edit_contrast(gray_scale, 5)

        # 加工した画像からフレームQRコードを取得してデコードする
        codes = decode(image)
        if len(codes) > 0:
            input=codes[0][0].decode('utf-8', 'ignore')
            num0=input #[1:len(input)-1]
            #num0=np.float(input[1:4])
            #num1=np.float(input[5:8])
            # コード内容を出力
            print(num0)
            #print(num0,num1,num0*num1)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame,input,(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
            img_pil = Image.fromarray(frame) # 配列の各値を8bit(1byte)整数型(0～255)をPIL Imageに変換。
            draw = ImageDraw.Draw(img_pil) # drawインスタンスを生成
            position = (50, 100) # テキスト表示位置
            draw.text(position, input, font = font , fill = (b, g, r, a) ) # drawにテキストを記載 fill:色 BGRA (RGB)
            img = np.array(img_pil) # PIL を配列に変換
            ## 表示
            cv2.imshow("res", img)
            cv2.imwrite("res.png", img)
            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) >= 0:
                break
    
        ret, frame = capture.read()        