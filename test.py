import numpy as np
import cv2
import os
import random
import tensorflow as tf

h,w = 48,152
num_cases = 16

images = []
labels = []

files = os.listdir('./dataset/images/')
random.shuffle(files)

model = tf.keras.models.load_model('my_model')

for f in files[0:num_cases]:
    test_img = cv2.imread('./dataset/images/' + f)
    resized_img = cv2.resize(test_img,(w,h))
    cropped_img = resized_img[h//2:]/255
    cropped_img = np.reshape(cropped_img,
        (1,cropped_img.shape[0],cropped_img.shape[1],cropped_img.shape[2]))

    test_out = model.predict(cropped_img)
    test_out = test_out[0,:,:,0]*128
    test_out = np.clip(test_out,0,128)
    
    resized_test_out = cv2.resize(test_out,(test_img.shape[1],test_img.shape[0]//2))
    resized_test_out = resized_test_out.astype(np.uint16)

    tih = test_img.shape[0]
    rih = resized_test_out.shape[0]

    test_img = test_img.astype(np.uint16)

    test_img[tih-rih:tih,:,1] = test_img[tih-rih:tih,:,1] + resized_test_out
    test_img = np.clip(test_img,0,255)

    test_img = test_img.astype(np.uint8)

    cv2.imshow('test_img',test_img)
    cv2.waitKey(0)
