from time import time
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from sklearn.ensemble import AdaBoostClassifier
import cv2


@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


def print_features(images, idx_sorted, feature_coord):
    idx_sorted = np.argsort(clf.feature_importances_)[::-1]
    fig, axes = plt.subplots(3, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = images[0]
        image = draw_haar_like_feature(image, 0, 0,
                                       images.shape[2],
                                       images.shape[1],
                                       [feature_coord[idx_sorted[idx]]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    _ = fig.suptitle('The most important features')


images = lfw_subset()


feature_types = None

X = delayed(extract_feature_image(img, feature_types) for img in images)

X = np.array(X.compute(scheduler='threads'))

y = np.array([1] * 100 + [0] * 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
                                                    random_state=0,
                                                    stratify=y)


feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)

clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=200, max_depth=None,
                                                               max_features=1000, n_jobs=-1, random_state=42),
                         n_estimators=200, random_state=42)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

idx_sorted = np.argsort(clf.feature_importances_)[::-1]

print_features(images, idx_sorted, feature_coord)


def resize(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return img


img = cv2.imread("face.jpg")
image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
image = resize(image, 0.3)


tmp = image.copy()
(w_width, w_height) = (25, 25)
all_windows = []
show_all = []
coords = []
for x in range(0, image.shape[1] - w_width, 5):
    for y in range(0, image.shape[0] - w_height, 5):
        window = image[x:x + w_width, y:y + w_height]
        if window.shape == (25, 25):
            coords.append((x, x + w_width, y, y + w_height))
            show_all.append(np.array(window))
            all_windows.append(np.array(window))

all_windows = np.array(all_windows)

feat = delayed(extract_feature_image(im, feature_types) for im in all_windows)
feat = np.array(feat.compute(scheduler='threads'))
pred = clf.predict_proba(feat)
result = list(zip(show_all, coords, pred))
result = np.array(list(filter(lambda x: x[2][1] > 0.85, result)))

x = min([i[2] for i in result[:, 1]])
x1 = max([i[3] for i in result[:, 1]])

y = min([i[0] for i in result[:, 1]])
y1 = max([i[1] for i in result[:, 1]])

img_res = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
img_res = cv2.rectangle(img_res.copy(), (x, y), (x1, y1), (255, 0, 0), 2)
cv2.imwrite("res.jpg", img_res)
