import cv2 as cv
import numpy as np
import csv

class Face(object):
    def __init__(self, score_threshold = 0.3, data_file='database.csv') -> None:

        self.csv_file = data_file
        self.score_threshold = score_threshold

        self.detector = cv.FaceDetectorYN.create(
            model="face_detection_yunet_2022mar.onnx",
            config="",
            #   input_size=[480, 640], # [width, height]
            input_size=[640, 480], # [width, height]
            score_threshold=0.99,
            backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
            target_id=cv.dnn.DNN_TARGET_CPU, # optional
            )

        self.recognizer = cv.FaceRecognizerSF.create(
            model="face_recognition_sface_2021dec.onnx",
            config="",
            backend_id=cv.dnn.DNN_BACKEND_DEFAULT, # optional
            target_id=cv.dnn.DNN_TARGET_CPU, # optional
            )

        self.user_buffer = []
        self.feature_buffer = []

        with open(self.csv_file,"r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if len(line)>0:
                    self.user_buffer.append(line[0]) #name, feature
                    self.feature_buffer.append(list(map(lambda x:float(x, ), line[1:])))
        print("Finished Initialization.")
    
    def regist(self, video_frame, user_name):
        faces = self.detector.detect(video_frame)[1]
        face = faces[0][:-1] # take the first face and filter out 
        aligned_face = self.recognizer.alignCrop(video_frame, face)
        feature = self.recognizer.feature(aligned_face)

        feature = feature.reshape((128,))
        feature = feature.tolist()
        # feature = list(map(lambda x:str(x), feature))

        if len(self.user_buffer)==0 or user_name not in self.user_buffer:
            self.user_buffer.append(user_name)
            self.feature_buffer.append(feature)
        else:
            index = self.user_buffer.index(user_name)
            self.user_buffer.pop(index)
            self.feature_buffer.pop(index)
            self.user_buffer.append(user_name)
            self.feature_buffer.append(feature)
        
        print("Finished Registration.")

    def delete(self, user_name):
        if user_name not in self.user_buffer:
            print("Warning: NO SUCH USER!!!")
        else:
            index = self.user_buffer.index(user_name)
            self.user_buffer.pop(index)
            self.feature_buffer.pop(index)
            print("Sussessfully Deleted.")
    
    def recognize(self, video_frame):
        faces = self.detector.detect(video_frame)[1]
        face = faces[0][:-1] # take the first face and filter out 
        aligned_face = self.recognizer.alignCrop(video_frame, face)
        feature = self.recognizer.feature(aligned_face)

        
        max_score = 0
        match_user = None

        if(len(self.feature_buffer))!=0:
            for user_name, featureB in zip(self.user_buffer, self.feature_buffer):
                b = np.array([featureB]).astype(np.float32)

                score = self.recognizer.match(feature, b)
                if score>self.score_threshold and score>max_score:
                    max_score=score
                    match_user=user_name
                
        
        if match_user is None:
            print("UnMatch!!!")
        else:
            print(match_user)
        
        return match_user


    def update(self):#每次结束系统都要做
        with open("database.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            for user_name, feature in zip(self.user_buffer, self.feature_buffer):
                feature = list(map(lambda x:str(x), feature))
                writer.writerow((user_name, *feature))




        
if __name__ == '__main__':
    # Step 2: load image
    img1 = cv.imread("3.png")
    img2 = cv.imread("4.png")
    img1 = cv.resize(img1, (640, 480))
    img2 = cv.resize(img2, (640, 480))

    face = Face()
    face.regist(img1, 'A')
    face.regist(img1, 'A')
    face.recognize(img2)
    # face.update()
    # face.recognize(img2)
    face.delete('A')
    face.update()#每个程序都要有

    # print(score)
    cv.waitKey()

    # score >= cosine similarity threshold, matched
    # score <= l2 distance threshold, matched
