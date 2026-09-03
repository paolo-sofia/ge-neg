import pathlib

import cv2
import numpy as np

from src.ge_neg.border_identifier import BorderIdentifier

path = pathlib.Path("/home/paolo/git/ge-neg/scans/img_060_02.tif")


if __name__ == "__main__":
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float: np.ndarray = img.astype(np.float32) / (2 ** (img.dtype.itemsize * 8) - 1)

    border_identifier = BorderIdentifier(img_float)
    border_identifier.find_borders()
