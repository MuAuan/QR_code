from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

s=10
#N=20
#RATE = 11025
#fr = RATE
#fn=51200*N/50  #*RATE/44100
#fs=fn/fr
#list=[0,0.2,0.5,1,2,5,10,20,50]
#list_=['00100','00500','01000','01500','02000','02500','03000','03500','04000','05000','05700','07000','08000','09000','10000']
list_=['[-1. -1.0]','[-1.  -0.9]','[-1.  -0.8]','[-1.  -0.7]','[-1.  -0.6]','[-1.  -0.5]','[-1.  -0.4]','[-1.  -0.3]','[-1.  -0.2]','[-1.  -0.1]']
images = []
s=0
#for i in range(0,20000,1000):
for i in list_:
    im = Image.open('./gen_images/qr_img_'+i+'.png') 
    im =im.resize(size=(640,640), resample=Image.NEAREST)  #- NEAREST - BOX - BILINEAR - HAMMING - BICUBIC - LANCZOS
    images.append(im)
    print(s,type(im),np.array(im).shape)
    s+=1
for i in range(s):
    plt.imshow(images[i])
    plt.pause(1)
    print(type(images[i]))
    
images[0].save('./gen_images/qr_img_.gif', save_all=True, append_images=images[1:], optimize=False, duration=100*20, loop=0)    
#C:\Users\MuAuan\example\gen_images\128x128_32x32_final
#C:\Users\MuAuan\example\keras-srgan\gen_images_cgan\generate0-100000
#C:\Users\MuAuan\example\keras-srgan\gen_images_cgan\cgan_gendiscri2D