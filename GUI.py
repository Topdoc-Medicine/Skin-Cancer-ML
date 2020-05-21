from Tkinter import *
import tkMessageBox as messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas
import h5py
import glob
from keras.initializers import glorot_uniform

h5file =  "weights2.h5"

with h5py.File(h5file,'r') as fid:
     model = load_model(fid)

def get_filenames():
    global path
    path = r"test"
    return os.listdir(path)

def autoroi(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]

    return roi


def prediction():
    list_of_files = glob.glob('./skin-cancer-mnist-ham10000/HAM10000_images_part_1/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(latest_file)
    # img = cv2.imread("./cell_images/cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_182.png")
    img = autoroi(img)
    img = cv2.resize(img, (75, 100))
    img = np.reshape(img, [1, 75, 100, 3])
    img = tf.cast(img, tf.float64)


    print(img)

    prediction = model.predict(img)
    prediction = prediction.argmax(axis=1)
    print(prediction)


    return(prediction)


finalPrediction = prediction()
print(finalPrediction[[0]])
if (finalPrediction == 0):
    print('You have been diagnosed with Melanocytic nevi. Please contact a doctor for assistance soon.')
elif (finalPrediction == 1):
    print('You have been diagnosed with Melanoma. Please contact a doctor for assistance soon.')
elif (finalPrediction == 2):
    print('You have been diagnosed with Benign keratosis-like lesions. Please contact a doctor for assistance soon.')
elif (finalPrediction == 3):
    print('You have been diagnosed with Basal Cell Carcinoma. Please contact a doctor for assistance soon.')
elif (finalPrediction == 4):
    print('You have been diagnosed with Actinic Keratoses. Please contact a doctor for assistance soon.')
elif (finalPrediction == 5):
    print('You have been diagnosed with Vascular Lesions. Please contact a doctor for assistance soon.')
elif (finalPrediction == 6):
    print('You have been diagnosed with Dermatofibroma. Please contact a doctor for assistance soon.')
