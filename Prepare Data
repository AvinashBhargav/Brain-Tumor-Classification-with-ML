from google.colab import drive
import os

drive.mount('/content/drive')
drive_path = '/content/drive/MyDrive/brain-tumor-detection-master/brain_tumor/Training/'

path_contents = os.listdir(drive_path)

classes = {'no_tumor': 0, 'pituitary_tumor': 1}

import cv2
X = []
Y = []
for cls in classes:
    pth = '/content/drive/MyDrive/brain-tumor-detection-master/brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])

X = np.array(X)
Y = np.array(Y)

np.unique(Y)

pd.Series(Y).value_counts()

X.shape
