import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

def toMatrix(imgobj):
    return np.array(imgobj)

def toGrayScale(rgb_data):
    gray_data = [
        list(map(
            lambda rgb: rgb[0]*0.299 + rgb[1]*0.578 + rgb[2]*0.114, rgb_data[line]
        ))
        for line in range(rgb_data.shape[0])
    ]
    return gray_data

def Binarize(img_gray, btype='adaptive_mean'):
    try:
        btype = ''.join(list(map(
            lambda c: c.capitalize(), btype.split('_')
        )))
    except:
        btype = btype.capitalize()
    finally:
        th = eval('calc%sTh'%btype)(img_gray)
    img_binarry = (np.array(img_gray)>th).astype(np.int)*255
    return img_binarry

def calcAdaptiveMeanTh(img_gray):
    return signal.correlate(
                img_gray, np.ones(9).reshape(3,3)/9, 'same'
                )

def calcOtsuTh(img_gray):
    img_gray = np.array(img_gray)
    gf_mean = np.mean(img_gray)
    st = np.floor(img_gray.min()).astype(int)
    ed = np.ceil(img_gray.max()).astype(int)
    smax = 0
    th = 0
    for n in range(st, ed):
        g1 = img_gray[img_gray<n]
        g2 = img_gray[img_gray>=n]
        s1 = np.var(g1)
        s2 = np.var(g2)
        n1 = len(g1)
        n2 = len(g2)
        sw = (n1*s1 + n2*s2)/(n1 + n2)
        sb = (n1*(np.mean(g1) - gf_mean)**2 + n1*(np.mean(g1) - gf_mean)**2)/(n1 + n2)   
        if smax < sb/sw:
            smax = sb/sw
            th = n
    return th

if __name__ == '__main__':
    # load image
    imgobj = Image.open('./img/original/asg1.png')

    # change the loaded image data from rgb to gray scale
    img_rgb = toMatrix(imgobj)
    img_gray = toGrayScale(img_rgb)

    # binalize gray scale data
    img_bin = Binarize(img_gray)

    # show image
    imgobj = Image.fromarray(np.uint8(img_bin))
    imgobj.show()