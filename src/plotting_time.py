import os
from matplotlib import figure
import cv2
import numpy as np


def create_image_dataset(df, path, dpi, img_width=432, img_height=288, target='target', verbose=True):
    X = df.drop(columns=[target]).values
    y = df[target].values
    if not os.path.isdir(path):
        os.makedirs(path)
    for i in range(len(X)):
        image_path = os.path.join(path, str(y[i]))
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        image_path = os.path.join(image_path, str(i) + '.png')
        fig = figure.Figure(figsize=(img_width / dpi, img_height / dpi))
        ax = fig.subplots(1)
        ax.plot(X[i])
        fig.savefig(image_path, dpi=dpi)
        if verbose:
            print(f'[{i+1}/{len(X)}] image generated')


def load_image_dataset(path, treshold=170):
    X = []
    y = []
    classes = os.listdir(path)
    for i, cls in enumerate(classes):
        dir_path = os.path.join(path, cls)
        images = os.listdir(dir_path)
        for img in images:
            y.append(i)
            img_array = cv2.imread(os.path.join(dir_path, img))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            thresh, img_array = cv2.threshold(
                img_array, treshold, 255, cv2.THRESH_BINARY)
            X.append(img_array)
    X = np.array(X)
    y = np.array(y)
    X[X == 0] = [1]
    X[X == 255] = [0]
    return X, y, classes
