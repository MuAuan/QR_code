from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

s=10
list_=['[-1. -1.0]','[-1.  -0.9]','[-1.  -0.8]','[-1.  -0.7]','[-1.  -0.6]','[-1.  -0.5]','[-1.  -0.4]','[-1.  -0.3]','[-1.  -0.2]','[-1.  -0.1]']
images = []
s=0

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
