import qrcode
from PIL import Image

img_bg = Image.open('./gen_images/generator_22000_1.png')

qr = qrcode.QRCode(box_size=4)
qr.add_data('I am Lena')
qr.make()
img_qr = qr.make_image()

pos = (img_bg.size[0] - img_qr.size[0], img_bg.size[1] - img_qr.size[1])

img_bg.paste(img_qr, pos)
img_bg.save('./gen_images/qr_lena.png')