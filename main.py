import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
import os
import cv2

location = "Data\\CroppedYale"

def loaddata(directory):
    datamatrix = []
    for subdir in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, subdir)):
            with Image.open(os.path.join(directory, subdir, filename)) as im:
                imarray = np.array(im)
                datamatrix.append(np.reshape(imarray, 32256))
    return np.array(datamatrix)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = np.transpose(loaddata(location))
    print(data)
    U, s, VT = svd(data)
    Eigenfaces = np.transpose(U)
    print(Eigenfaces)
    print(s)
    print(U.shape)
    for count,ar in enumerate(Eigenfaces):
        im = cv2.cvtColor(np.reshape(ar, (192, 168)),cv2.COLOR_GRAY2RGB) * s[count]
        cv2.imshow("Image", im)
        cv2.imwrite(str(count) + '.png', im)
        if count > 20:
            break


# np.reshape(arrayvec, (192, 168))


