# from Tkinter import *
# import tkMessageBox as messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas
import h5py
import glob
import torch
from torchvision import models
from keras.initializers import glorot_uniform


h5file =  "weights.h5"
# model = torch.load(h5file)
# model.load_state_dict(checkpoint)
# # model = model.load_state_dict(torch.load(h5file))
# # with h5py.File(h5file,'r') as fid:
# #      model = load_model(fid)

# model = models.densenet121()
# state = torch.load(h5file)
# model.load_state_dict(state)

model = models.densenet121(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
input_size = 224

model = torch.load(h5file)



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
    list_of_files = glob.glob('./skin-cancer-mnist-ham10000/ham10000_images_part_1/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(latest_file)
    img = autoroi(img)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    img = tf.cast(img, tf.float64)
    img = tf.Tensor(img, (), int32)
    print(img)
    outputs = model(img)
    prediction = torch.argmax(outputs, 1)
    # prediction = model.predict(img)
    # Class = prob.argmax(axis=1)
    print(prediction)


    return(prediction)


finalPrediction = prediction()
print("Final Prediction = ", finalPrediction)
# if (finalPrediction == 0):
#     print("Congratulations! You are healthy!")
# else:
#     print("Unfortunately, you have been diagnosed with Malaria.")

if finalPrediction == 'tensor(0)':
    ret = 'POSITIVE FOR SKIN CANCER: Melanocytic nevi'
elif finalPrediction == 'tensor(1)' or finalPrediction == 'tensor(6)':
    ret = 'POSITIVE FOR SKIN CANCER: Dermatofibroma'
elif finalPrediction == 'tensor(2)':
    ret = 'NEGATIVE FOR SKIN CANCER: Benign keratosis-like lesions'
elif finalPrediction == 'tensor(3)':
    ret = 'POSITIVE FOR SKIN CANCER: Basal cell carcinoma'
elif finalPrediction == 'tensor(4)':
    ret = 'POSITIVE FOR SKIN CANCER: Actinic keratoses'
elif finalPrediction == 'tensor(5)':
    ret = 'POSITIVE FOR SKIN CANCER: Vascular lesions'
