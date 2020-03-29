import cv2
import numpy as np


def sobel_op(img):  # оператор Собеля
    # сопоставление каждой точке вектора градиента
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):  # подавление не максимумов
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q, r = 255, 255
            # 0 градусов
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            # 45 градусов
            elif (22.5 <= angle[i, j] < 67.5):
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            # 90 градусов
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i + 1, j]
                r = img[i - 1, j]
            # 135 градусов
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z


def threshold(img, low_tresh=0.05, high_thresh=0.09, weak=20):
    # пороговая фильтрация
    img = img.copy()
    highThreshold = img.max() * high_thresh
    lowThreshold = highThreshold * low_tresh
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(weak)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def ambiguity_trace(img, weak, strong=255):
    # операции над пикселями, получившими промежуточное значение на предыдущем этапе

    M, N = img.shape
    r_img = img.copy()
    move = [[-1, -1, -1, 0, 0, 1, 1, 1],
            [-1, 0, 1, -1, 1, -1, 0, 1]]
    # если пиксели соединены с установленной границей, то их отсносим к границе
    for i in range(0, M):
        for j in range(0, N):
            if img[i, j] == strong:
                r_img[i, j] = strong
                for k in range(8):  # пиксель относим к границе, если он соприкасается с ней по одному из направлений
                    dx = move[0][k]
                    dy = move[1][k]
                    X = i
                    Y = j
                    while True:
                        X += dx
                        Y += dy
                        if X < 0 or Y < 0 or X > M - 1 or Y > N - 1:
                            break
                        if img[X, Y] == 0 or img[X, Y] == strong:
                            break
                        r_img[X, Y] = strong
            else:
                r_img[i, j] = 0

    return img


def edge_detector(image, thresh_min, thresh_max):
    blured = cv2.GaussianBlur(image, (7, 7), 1)  # изображения сглаживается фильром Гаусса
    grad, theta = sobel_op(blured)  # применяется оператор Собеля
    suppressed = non_max_suppression(grad, theta)  # происходит подавление не максимумов
    threshed, weak, strong = threshold(suppressed, thresh_min, thresh_max)  # применяется пороговая фильтрация
    res_img = ambiguity_trace(threshed, weak)  # операции над неоднозначными пикселями
    return res_img


img = cv2.imread("face.jpg", 0)
res = edge_detector(img, 0.06, 0.09)
cv2.imwrite("res.jpg", res)
