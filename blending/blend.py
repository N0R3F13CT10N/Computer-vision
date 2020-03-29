import cv2
import numpy as np


def gauss_pyramid(img, depth):
    pyramid = [img]
    for i in range(depth):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def laplacian_pyramid(gauss_pyr, depth):
    lapl_pyr = [gauss_pyr[depth - 1]]
    for i in range(depth - 1, 0, -1):
        size = (gauss_pyr[i - 1].shape[1], gauss_pyr[i - 1].shape[0])
        G = cv2.pyrUp(gauss_pyr[i], dstsize=size)
        L = np.subtract(gauss_pyr[i - 1], G)
        lapl_pyr.append(L)
    return lapl_pyr


def restore_image(src):
    res = src[0]
    for i in range(1, 5):
        g_s = src[i].shape
        res = cv2.pyrUp(res)[:g_s[0], :g_s[1], :]
        res = cv2.add(res, src[i])
    return res


def blending(orig, eye_mask, emptyt_mask, depth):
    # раскладываем оригинал и маски по пирамидам Гаусса и Лапласа
    gpA = gauss_pyramid(eye_mask.copy(), depth)
    gpB = gauss_pyramid(orig.copy(), depth)
    gpM = gauss_pyramid(emptyt_mask.copy(), depth)

    lpA = laplacian_pyramid(gpA, depth)
    lpB = laplacian_pyramid(gpB, depth)
    lpM = laplacian_pyramid(gpM, depth)

    gpM = gpM[:-1]

    res = []
    for la, lb, lm in zip(lpA, lpB, gpM[::-1]):  # применяем блендинг согласно формуле
        ls = la * lm + lb * (1.0 - lm)
        res.append(ls)

    cv2.imwrite("res.jpg",  restore_image(res))


orig = cv2.imread('face.jpg').astype("float64")
eye_left, eye_right = 100, 300  # диапазон копируемого глаза по оси x
eye_top, eye_bottom = 210, 300  # диапазон копируемого глаза по оси y
frame_v, frame_h = 30, 30  # рамка по x и y
forehead_left, forehead_top = 221, 30  # левый верхний угол прямоугольника с рамкой для вставки во лбу
eye_forehead_left, eye_forehead_top = 267, 60  # он же, без учёта рамки

eye = orig[eye_top:eye_bottom, eye_left:eye_right].copy()  # копируем глаз
cv2.imwrite("sample0.jpg", eye)


eye_framed = orig[eye_top - frame_v:eye_bottom + frame_v,
               eye_left - frame_h:eye_right + frame_h].copy()  # копируем глаз с рамкой
cv2.imwrite("sample1.jpg", eye_framed)

eye_mask = np.zeros(orig.shape, dtype='float32')
eye_mask[forehead_top:forehead_top + eye_framed.shape[0],
         forehead_left:forehead_left + eye_framed.shape[1], :] = eye_framed
cv2.imwrite("sample2.jpg", eye_mask)  # маска глаза во лбу


empty_mask = np.zeros(orig.shape, dtype='float32')  # белая маска во лбу без рамки
empty_mask[eye_forehead_top: eye_forehead_top + eye.shape[0],
           eye_forehead_left:eye_forehead_left + eye.shape[1], :] = (1, 1, 1)

blending(orig, eye_mask, empty_mask, 5)
