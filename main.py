#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
from PIL import Image
from Sec_rainbow import *
from First_fresnel import *
from process import process
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

def amp_holo(image_path):
    g = Image.open(image_path, mode="r")
    # convert RGB image to Gray image
    g = g.convert("L")
    g1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, g1 = cv2.threshold(g1, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('raw', g1)
    G = np.array(g)
    cv2.waitKey(0)
    # flag=0: 球面波
    # flag=1: 平面波
    # 其他: 物体相位
    fresnel = Fresnel(flag=0)
    # 全息记录
    holo_first_amp, conjwave = fresnel.amp_record(G)
    holo_first_amp_show = Image.fromarray(fresnel.format_img(holo_first_amp)).convert("L")
    holo_first_amp_show.save("holo_first_amp.bmp", "bmp")
    holo_first_amp_show = cv2.imread("holo_first_amp.bmp", cv2.IMREAD_GRAYSCALE)
    holo_first_amp_show = cv2.applyColorMap(holo_first_amp_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('first_fresnel_hologram_amp', holo_first_amp_show)
    cv2.waitKey(0)

    # 二步加狭缝
    new_img_amp = holo_first_amp_show
    w = new_img_amp.shape[1]
    h = new_img_amp.shape[0]
    new_img_amp[0:100, 0:w] = 0
    new_img_amp[180:h, 0:w] = 0
    cv2.imshow('H1', new_img_amp)
    cv2.imwrite('holo1_amp.bmp', new_img_amp)
    cv2.waitKey(0)
    # 全息再现为赝像
    holo1_amp = Image.open('holo1_amp.bmp', mode="r")
    holo1_amp = holo1_amp.convert("L")
    H1_amp = np.array(holo1_amp)
    fake_amp = fresnel.reconstruct(H1_amp, 1)
    fake_amp = fresnel.format_img(fake_amp)
    fake_amp_show = Image.fromarray(fake_amp).convert("L")
    fake_amp_show.save("fake_amp.bmp", "bmp")
    fake_amp_show = cv2.imread("fake_amp.bmp", cv2.IMREAD_GRAYSCALE)
    fake_amp_show = cv2.applyColorMap(fake_amp_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('fake_amp', fake_amp_show)
    cv2.waitKey(0)

    # 记录赝像为H2
    erbu = Erbu(flag=0)
    fake_amp = Image.open('fake_amp.bmp', mode="r")
    F = fake_amp.convert('L')
    f = np.array(F)
    H2_amp , conjwave= erbu.amp_record(f)
    holo_sec_amp = fresnel.format_img(H2_amp)
    print(H2_amp.shape, H2_amp)
    holo_sec_amp_show = Image.fromarray(holo_sec_amp).convert("L")
    holo_sec_amp_show.save("holo2_amp.bmp", "bmp")
    holo_sec_amp_show = cv2.imread("holo2_amp.bmp", cv2.IMREAD_GRAYSCALE)
    holo_sec_amp_show = cv2.applyColorMap(holo_sec_amp_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('sec_rainbow_holo_amp', holo_sec_amp_show)
    cv2.waitKey(0)

    #重建
    holo2_amp = Image.open('holo2_amp.bmp', mode="r")
    holo2_amp = holo2_amp.convert("L")
    H2_amp = np.array(holo2_amp)
    rec= erbu.reconstruct(H2_amp, conjwave)
    rec = fresnel.format_img(rec)
    rec_img = Image.fromarray(rec).convert("L")
    rec_img.save('rec_amp.bmp', 'bmp')
    rec_img = cv2.imread('rec_amp.bmp', cv2.IMREAD_GRAYSCALE)
    rec_img = cv2.applyColorMap(rec_img, cv2.COLORMAP_RAINBOW)
    cv2.imshow('rec', rec_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.waitKey(delay=0)

    cv2.destroyAllWindows()
    '''ret, final = cv2.threshold(rec_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('raw', final)
    cv2.waitKey(0)'''


def pha_holo(image_path):
    g = Image.open(image_path, mode="r")
    # convert RGB image to Gray image
    g = g.convert("L")
    g1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, g1 = cv2.threshold(g1, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('raw', g1)
    G = np.array(g)
    cv2.waitKey(0)
    # flag=0: 球面波
    # flag=1: 平面波
    # 其他: 物体相位'''
    fresnel = Fresnel(flag=0)
    # 全息记录
    holo_first_pha = fresnel.pha_record(G)
    print(holo_first_pha.shape, holo_first_pha)
    holo_first_pha_show = Image.fromarray(fresnel.format_img(holo_first_pha)).convert("L")
    holo_first_pha_show.save("holo_first_pha.bmp", "bmp")
    holo_first_pha_show = cv2.imread("holo_first_pha.bmp", cv2.IMREAD_GRAYSCALE)
    holo_first_pha_show = cv2.applyColorMap(holo_first_pha_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('first_fresnel_hologram_pha', holo_first_pha_show)
    cv2.waitKey(0)
   
    # 二步加狭缝
    new_img_pha = holo_first_pha
    erbu = Erbu(flag=0)
    new_img_pha[0:160, 0:new_img_pha.shape[1]] = 0
    new_img_pha[250:new_img_pha.shape[0], 0:new_img_pha.shape[1]] = 0
    new_img_pha_show = Image.fromarray(erbu.format_img(new_img_pha)).convert("L")
    new_img_pha_show.save("holo1_pha.bmp", "bmp")
    new_img_pha_show = cv2.imread("holo1_pha.bmp", cv2.IMREAD_GRAYSCALE)
    new_img_pha_show = cv2.applyColorMap(new_img_pha_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('H1', new_img_pha_show)
    cv2.waitKey(0)
    # 全息再现为赝像
    '''holo1_pha = Image.open('holo1_pha.bmp', mode="r")
    holo1_pha = holo1_pha.convert("L")
    H1_amp = np.array(holo1_pha)
    fake_pha = fresnel.reconstruct(H1_amp)'''
    fake_pha = fresnel.reconstruct(new_img_pha)
    fake_pha1 = erbu.format_img(fake_pha)
    print(fake_pha.shape, fake_pha)
    fake_pha_show = Image.fromarray(fake_pha1).convert("L")
    fake_pha_show.save("fake_pha.bmp", "bmp")
    fake_pha_show = cv2.imread("fake_pha.bmp", cv2.IMREAD_GRAYSCALE)
    fake_pha_show = cv2.applyColorMap(fake_pha_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('fake_pha', fake_pha_show)
    cv2.waitKey(0)

    # 记录赝像为H2

    '''fake_pha = Image.open('fake_pha.bmp', mode="r")
    F = fake_pha.convert('L')
    f = np.array(F)
    H2_pha = erbu.pha_record(f)'''
    H2_pha = erbu.pha_record(fake_pha)
    holo_sec_pha = erbu.format_img(H2_pha)
    print(H2_pha.shape, H2_pha)
    holo_sec_pha_show = Image.fromarray(holo_sec_pha).convert("L")
    holo_sec_pha_show.save("holo2_pha.bmp", "bmp")
    holo_sec_pha_show = cv2.imread("holo2_pha.bmp", cv2.IMREAD_GRAYSCALE)
    holo_sec_pha_show = cv2.applyColorMap(holo_sec_pha_show, cv2.COLORMAP_RAINBOW)
    cv2.imshow('sec_rainbow_holo_pha', holo_sec_pha_show)
    cv2.waitKey(0)

    #重建
    '''holo2_pha = Image.open('holo2_pha.bmp', mode="r")
    holo2_pha = holo2_pha.convert("L")
    H2_pha = np.array(holo2_pha)'''
    rec= erbu.reconstruct(H2_pha)
    rec = erbu.format_img(rec)
    rec_img = Image.fromarray(rec).convert("L")
    rec_img.save('rec_pha.bmp', 'bmp')
    rec_img = cv2.imread('rec_pha.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('rec', rec_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # input the image path
    image_path = "1.jpg"
    #image_path = "../Res/Set5/GTmod12/butterfly.png"
    amp_holo(image_path)
    # pha_holo(image_path)


