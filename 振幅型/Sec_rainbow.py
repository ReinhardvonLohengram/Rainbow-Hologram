import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2
import random
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

class Erbu: # z0是物体到平板距离，zr是
    def __init__(self, lambda_=671 * nm, pix=8 * um, z0=1500 * mm, zr=1500 * mm, flag=0, shift=0):
        # 默认红光
        self.lambda_ = lambda_
        self.pix = pix
        self.z0 = z0
        self.zr = zr
        self.flag = flag
        self.shift = shift
# 定义物理量
    def zero_padding(self, raw_img):
        height, width = raw_img.shape # raw_img为一个矩阵
        w = 2 * width
        h = 2 * height
        pad_img = np.zeros((w, h))
        posx = (w - width) // 2
        posy = (h - height) // 2
        pad_img[posy:posy + height, posx:posx + width] = raw_img[:, :] / 255
        # 将raw_img的值变为0~1，储存在pad_img中央
        return pad_img
# 生成全息图的平板raw_img
    def fresnel(self, pad_img, conjwave, flag=True):  # 菲涅尔衍射
        h, w = pad_img.shape
        Lx0 = np.sqrt(self.lambda_ * self.z0 * w)  # FFT计算时同时满足振幅及相位取样条件的物光场宽度
        Ly0 = np.sqrt(self.lambda_ * self.z0 * h)
        if flag:
            dx0 = Lx0 / w
            dy0 = Ly0 / h
        else:
            dx0 = self.pix
            dy0 = self.pix

        # 菲涅尔全息计算
        x = dx0 * np.array(range(-w // 2, w // 2))
        y = dy0 * np.array(range(-h // 2, h // 2))
        X, Y = np.meshgrid(x, y)
        Fresnel = np.exp(1j * np.pi * (np.power(X, 2) + np.power(Y, 2)) / (self.lambda_ * self.z0))
        f2 = pad_img * Fresnel * conjwave
        f2 = fftshift(f2)
        Uf = fft2(f2)
        Uf = fftshift(Uf)
        dx1 = (self.lambda_ * self.z0) / (w * dx0)
        dy1 = (self.lambda_ * self.z0) / (h * dy0)
        x1 = dx1 * np.array(range(-w // 2, w // 2))
        y1 = dy1 * np.array(range(-h // 2, h // 2))
        X1, Y1 = np.meshgrid(x1, y1)
        '''L = -w * dx1
        xr = -L / 2
        Fresnel = np.exp(1j * np.pi * (np.power(X + xr, 2) + np.power(Y + xr, 2)) / (
                self.lambda_ * 1.4 * xr) )  # 发散光
        f2 = pad_img * Fresnel
        Uf = fft2(f2)
        #Uf = fftshift(Uf)# 振幅不用要注释掉'''
        phase = np.exp(1j * 2 * np.pi * self.z0 / self.lambda_) / (1j * self.lambda_ * self.z0) * np.exp(
            1j * np.pi / self.lambda_ / self.z0 * (np.power(X1, 2) + np.power(Y1, 2)))
        Uf = Uf * phase
        T = dx0  # 空域取样间隔
        Uf = Uf * T * T  # 二维离散变换量值补偿
        return Uf
    def wave(self, pad_img):
        h, w = pad_img.shape
        Lx0 = np.sqrt(self.lambda_ * self.z0 * w)  # FFT计算时同时满足振幅及相位取样条件的物光场宽度
        Ly0 = np.sqrt(self.lambda_ * self.z0 * h)
        dx0 = Lx0 / w
        dy0 = Ly0 / h

        x0 = dx0 * np.array(range(-w // 2, w// 2))
        y0 = dy0 * np.array(range(-h // 2, h // 2))
        Y0, X0 = np.meshgrid(y0, x0)

        dx1 = (self.lambda_ * self.zr) / (w * dx0)
        dy1 = (self.lambda_ * self.zr) / (h * dy0)
        x1 = dx1 * np.array(range(-w // 2, w // 2))
        y1 = dy1 * np.array(range(-h // 2, h // 2))
        X1, Y1 = np.meshgrid(x1, y1)

        # 相位物体
        p = np.sin(Y0 * np.pi / 50)
        phaseObject = np.exp(1j * p)

        # planeWave reference
        # pr = 2 * np.pi * X1 * np.sin(5 * np.pi / 180) / self.lambda_ + self.shift
        phi = 2 * np.pi * X1 * np.sin(np.pi / 2) / self.lambda_ + self.shift
        planeWave = np.exp(1j * phi)

        # sphericalWave 会聚光
        L = -w * dx1
        xr = -L / 2
        phi = np.pi * ((np.power(X1 + xr, 2) + np.power(Y1 + xr, 2)) / (self.lambda_ * 1.4 * xr) + 2.8 * xr / self.lambda_)
        sphericalWave = np.exp(-1j * phi)  / 1.4

        if self.flag == 1:
            wave = planeWave
        elif self.flag == 0:
            wave = sphericalWave
        else:
            wave = phaseObject
        return wave
# 生成参考光wave
    def amp_record(self, raw_img, flag=True):
        # flag为True记录强度，否则记录振幅
        pad_img = self.zero_padding(raw_img)
        object_ = pad_img
        refer_ = self.wave(pad_img)
        # 干涉
        w1 = object_ + 10*  refer_
        '''t = 0.6 * np.power(np.abs(w1), 2) # 复振幅透射率系数_与光强相关的系数-实数
        w1 = np.multiply(np.power(np.abs(w1), 2), np.power(t, 2)) if flag else np.abs(w1)  # 求出全息平面上的强度分布
        '''
        w1 = np.abs(w1)
        conjwave =10* np.conj(refer_)
        return w1, conjwave

    def pha_record(self, raw_img, flag=True):
        # flag为True记录强度，否则记录振幅
        pad_img = self.zero_padding(raw_img)
        object_ = pad_img
        refer_ = self.wave(pad_img)
        # 干涉
        w1 = object_ +  0.5 * refer_
        '''t = 0.6 * np.exp(1j* 0.6 *np.power(np.abs(w1), 2))# 复振幅透射率系数_与光强相关的系数-复指数
        w1 = np.multiply(np.power(np.abs(w1), 2), np.power(t ,2)) if flag else np.abs(w1)  # 求出全息平面上的强度分布
        '''
        w1 = np.power(w1, 2)
        return w1

    def reconstruct(self, holo_img, conjwave, RoomNo=10):
        # image processing
        holo_img = holo_img - np.mean(holo_img)
        # 注意这里衍射因为用了条纹，使用了参考光才形成的，因此不用显式表达
        U0 = self.fresnel(holo_img, conjwave, False)
        # 弥补AOH衍射效率底的问题
        Gmax = np.max(np.abs(U0))
        Gmin = np.min(np.abs(U0))
        U1 = np.abs(U0)
        U1 = (U1 - Gmin) / (Gmax / RoomNo - Gmin)
        return U1
# 根据holo_img重建为U1
    def format_img(self, img):
        # img = img ** 0.5
        format_img = img * 255
        format_img = format_img.astype(np.uint8)
        return format_img
