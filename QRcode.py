# -*- coding: UTF-8 -*-

from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np

# 画像ファイルの指定
#image = "test.png"
image = "qrcode.png"

# QRコードの読取り
data = decode(Image.open(image))
input=data[0][0].decode('utf-8', 'ignore')

num0=np.float(input[1:4])
num1=np.float(input[5:8])
# コード内容を出力
print(num0,num1,num0*num1)