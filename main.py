from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap # icon and load image
from PyQt5.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # load image

import cv2 #open-cv use for image resize

import keras # keras tensorflow
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D