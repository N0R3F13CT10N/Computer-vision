import cv2
from matplotlib import pyplot


def show_diagram(image, array):
    for i, color in enumerate(array):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        pyplot.plot(hist, color=color)
        pyplot.xlim([0, 256])
    pyplot.show()


img = cv2.imread('test.jpg')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
show_diagram(lab, ["b"])
lab = lab.astype('float64')
L, A, B = cv2.split(lab)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(L)
ln = cv2.subtract(L, minVal)
ln = cv2.divide(ln, maxVal - minVal)
ln = cv2.multiply(ln, 255)
res = cv2.merge((ln, A, B))
res = res.astype('uint8')
show_diagram(res, ["b"])
final = cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
cv2.imwrite("res.jpg", final)
