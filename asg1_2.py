import pathlib
import numpy as np
from scipy import signal
from PIL import Image

class Binarization():
    def __init__(self, img_path):
        # load image
        self.img_original = Image.open(img_path)

        # change the loaded image data from rgb to gray scale
        self.img_rgb = self.toMatrix(self.img_original)
        self.img_gray = self.toGrayScale(self.img_rgb)
    
    def toMatrix(self, imgobj):
        return np.array(imgobj)

    def toGrayScale(self, rgb_data):
        return [
            list(map(
                lambda rgb: rgb[0]*0.299 + rgb[1]*0.578 + rgb[2]*0.114, rgb_data[line]
            ))
            for line in range(rgb_data.shape[0])
        ]

    def show(self, btype):
        if btype == 'original':
            self.img_original.show()
        else:
            self.img_bin = self.binarize(btype)
            self.img_bin.show()
    
    def binarize(self, btype):
        # binarize gray scale data
        th = self.getThreshold(btype)
        img_bin = (np.array(self.img_gray)>th).astype(np.int)*255

        #rendar binarized data to Image object
        return Image.fromarray(np.uint8(img_bin))

    def getThreshold(self, btype):
        try:
            btype = ''.join(list(map(
                lambda c: c.capitalize(), btype.split('_')
            )))
        except:
            btype = btype.capitalize()
        return getattr(self, 'calc%sTh'%btype)(self.img_gray)

    def calcAdaptiveMeanTh(self, img_gray):
        return signal.correlate(
                    img_gray, np.ones(9).reshape(3,3)/9, 'same'
                    )

    def calcOtsuTh(self, img_gray):
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

    def save(self, path):
        self.img_bin.save(path)

if __name__ == '__main__':
    bn = Binarization('./img/original/asg1.png')
    save_dir = pathlib.Path('./img/processed')
    btypes = ['adaptive_mean', 'otsu']

    for bt in btypes:
        bn.show(bt)
        bn.save(save_dir/'asg1_2_{}.png'.format(bt))