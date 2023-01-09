# import cv2 as cv
# import numpy as np

# net = cv.dnn.readNet()

# net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# img = cv.imread('1.jpg')
# blob = cv.dnn.blobFromImage(img)
# net.setInput(blob)

# output = net.forward()

import numpy as np
import cv2 as cv
import csv

# Step 1: load model
yunet = cv.FaceDetectorYN.create(
  model="face_detection_yunet_2022mar.onnx",
  config="",
#   input_size=[480, 640], # [width, height]
  input_size=[640, 480], # [width, height]
  score_threshold=0.99,
  backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
  target_id=cv.dnn.DNN_TARGET_CPU, # optional
)
sface = cv.FaceRecognizerSF.create(
  model="face_recognition_sface_2021dec.onnx",
  config="",
  backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
  target_id=cv.dnn.DNN_TARGET_CPU, # optional
)

# Step 2: load image
img1 = cv.imread("3.png")
img2 = cv.imread("4.png")
# img1 = cv.resize(img1, (480, 640))
# img2 = cv.resize(img2, (480, 640))
img1 = cv.resize(img1, (640, 480))
img2 = cv.resize(img2, (640, 480))

# cv.imshow('1',img1)
# cv.imshow('2',img2)

# Step 3: detect faces, align, extract features and match
faces1 = yunet.detect(img1)[1]
face1 = faces1[0][:-1] # take the first face and filter out score
faces2 = yunet.detect(img2)[1]
face2 = faces2[0][:-1]
aligned_face1 = sface.alignCrop(img1, face1)
aligned_face2 = sface.alignCrop(img2, face2)
feature1 = sface.feature(aligned_face1)
feature2 = sface.feature(aligned_face2)
score = sface.match(feature1, feature2)

feature1 = feature1.reshape((128,))
feature1 = feature1.tolist()
feature1 = list(map(lambda x:str(x), feature1))
with open("database.csv","a+") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(("aaa", *feature1))

with open("database.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        if len(line)>0:
            name = line[0]
            feature = list(map(lambda x:float(x), line[1:]))
            print(name,  feature)
print(score)
cv.waitKey()

# score >= cosine similarity threshold, matched
# score <= l2 distance threshold, matched
