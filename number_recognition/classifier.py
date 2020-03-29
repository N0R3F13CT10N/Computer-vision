# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# считываем модель
clf, pipe = joblib.load("digits_classifier.pkl")
img = cv2.imread("numbers.png")
# конвертим в GRAY и применяем фильтр Гаусса
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey = cv2.GaussianBlur(img_grey, (5, 5), 0)
# прменяем пороговую фильтрацию
_, img_tresh = cv2.threshold(img_grey, 90, 255, cv2.THRESH_BINARY_INV)

# ищем контуры
contours, _ = cv2.findContours(img_tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# для каждого контура готовим прямоугольник
rectangles = [cv2.boundingRect(ctr) for ctr in contours]

for x in rectangles:
    # вычисляем регион прямоугольника
    lgth = int(x[3] * 1.6)
    pt1 = int(x[1] + x[3] // 2 - lgth // 2)
    pt2 = int(x[0] + x[2] // 2 - lgth // 2)
    img_resized = img_tresh[pt1:pt1 + lgth, pt2:pt2 + lgth]
    img_resized = cv2.resize(img_resized, (28, 28), interpolation=cv2.INTER_AREA)
    img_resized = cv2.dilate(img_resized, (3, 3))
    # вычисляем гистограмму ориентированных градиентов
    roi_hog_fd = hog(img_resized, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    roi_hog_fd = pipe.transform(np.array([roi_hog_fd], 'float64'))
    nbr = clf.predict(roi_hog_fd)  # определяем число
    cv2.putText(img, str(int(nbr[0])), (x[0], x[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

cv2.imwrite("res.jpg", img)
