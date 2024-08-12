import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image


PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

# image must be preprocess to clear noise before using this augment
def RotateCentroidCharacter(img, angle = 180, angle_multiply = 1.5):
    numpy_image = numpy.array(img)  
    
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    # Threshold the image
    _, thresh = cv2.threshold(opencv_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Filter and merge small components
    min_area = 170  # Define a threshold area for small components
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            # Calculate distances to other centroids and find the nearest larger component
            distances = distance.cdist([centroids[i]], centroids[1:], 'euclidean')[0]
            valid_distances = distances[stats[1:, cv2.CC_STAT_AREA] >= min_area]
            if len(valid_distances) > 0:
                nearest_larger = np.argmin(valid_distances)
                labels[labels == i] = nearest_larger + 1
    
    # Recalculate centroids
    new_centroids = np.array([np.mean(np.argwhere(labels == i), axis=0) for i in range(1, num_labels) if np.any(labels == i)])
    
    # Fit a line through the centroids
    X = new_centroids[:, 1].reshape(-1, 1)  # X coordinates (columns)
    y = new_centroids[:, 0]  # y coordinates (rows)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    
    # Calculate the angle of rotation
    angle = np.arctan(slope) * (angle / np.pi)
    angle *= angle_multiply
    
    # Calculate the center of the image for rotation
    center = (thresh.shape[1] // 3, thresh.shape[0] // 3)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Perform the affine transformation (rotation)
    rotated_image = cv2.warpAffine(thresh, rotation_matrix, (thresh.shape[1], thresh.shape[0]), flags=cv2.INTER_LINEAR)
    color_converted = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_converted)


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        # img = CutoutAbs(img, int(32*0.5))
        return img

class AugmentData():
    normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    def __init__(self, height = 64, width = 128):
        self.weak = transforms.Compose([
                    transforms.RandomCrop(size=(height, width),
                                        padding=(8, 16),
                                        padding_mode='reflect')])

        self.strong = transforms.Compose([
                    transforms.RandomCrop(size=(height, width),
                                        padding=(8, 16),
                                        padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10)])

        self.MT = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomCrop(size=(height, width),
                                    padding=(8, 16),
                                    padding_mode='reflect'),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ])
