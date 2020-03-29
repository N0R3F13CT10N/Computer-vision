import cv2

img_orig = cv2.imread('contours/contours0.jpg')
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
img = cv2.GaussianBlur(img, (7, 7), 2)

img = img[:, :, 0]
cv2.imwrite("contours/contours1.jpg", img)
tresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("contours/contours2.jpg", img)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_res = cv2.drawContours(img_orig, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 2)
cv2.imwrite("contours/contours3.jpg", img_res)
print(sorted(list(map(lambda x: cv2.arcLength(x, True), contours))))
print(len(contours))


# img = cv2.imread('contours/contours0.jpg')
# img_canny = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_canny = cv2.GaussianBlur(img_canny, (7, 7), 1)
#
# img_canny = cv2.Canny(img_canny, 245, 250)
# contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 3)
# cv2.imwrite('contours/contoursCanny1.jpg', img)
# sorted(list(map(lambda x: cv2.arcLength(x, True), contours)))
