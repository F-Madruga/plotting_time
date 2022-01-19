from email.mime import image
import pandas as pd
from scipy.io import arff
from plotting_time import create_image_dataset, load_image_dataset
from PIL import Image


def load_arff_file(path):
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


if __name__ == '__main__':
    # df = load_arff_file('/home/madruga/git/plotting_time/ArrowHead_TEST.arff')
    # create_image_dataset(df, './img', 96)
    X, y, classes = load_image_dataset('./img')
    img = Image.fromarray(X[0] * 255)
    img.show()
