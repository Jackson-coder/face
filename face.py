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
        max_score = 0
        match_user = None

        faces = self.detector.detect(video_frame)[1]
        if faces is None:
            return match_user
        face = faces[0][:-1] # take the first face and filter out 
        aligned_face = self.recognizer.alignCrop(video_frame, face)
        feature = self.recognizer.feature(aligned_face)


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
            self.draw_face_rectangles(face, match_user, video_frame)
            print(match_user)
        
        return match_user

    def update(self):#每次结束系统都要做
        with open("database.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            for user_name, feature in zip(self.user_buffer, self.feature_buffer):
                feature = list(map(lambda x:str(x), feature))
                writer.writerow((user_name, *feature))

    def draw_face_rectangles(self, face, name, image):
        # Get the coordinates of the face
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        # Draw a rectangle around the face
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Get the size of the font to use for the name
        font = cv.FONT_HERSHEY_SIMPLEX
        # Get the size of the text
        text_size = cv.getTextSize(name, font, 1, 2)[0]
        # Calculate the position of the text
        text_x = x + w // 2 - text_size[0] // 2
        text_y = y - 20
        # Draw the text on the image
        cv.putText(image, name, (text_x, text_y), font, 1, (255, 0, 0), 2)
        return image

    def put_mask(self, image, RegistName=None, DeleteName=None):
        # 获取标签
        # 标签格式　bbox = [xl, yl, xr, yr]
        bbox1 = [20,20,180,60]
        bbox2 = [20,80,180,120]

        # 画出mask
        zeros1 = np.zeros((image.shape), dtype=np.uint8)
        zeros2 = np.zeros((image.shape), dtype=np.uint8)

        zeros_mask1 = cv.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                        color=(0,0,255), thickness=-1 ) #thickness=-1 表示矩形框内颜色填充
        zeros_mask2 = cv.rectangle(zeros2, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]),
                        color=(0, 255, 0), thickness=-1)

        zeros_mask = np.array((zeros_mask1 + zeros_mask2))

        try:
            # alpha 为第一张图片的透明度
            alpha = 1
            # beta 为第二张图片的透明度
            beta = 0.5
            gamma = 0
            # cv2.addWeighted 将原始图片与 mask 融合
            mask_img = cv.addWeighted(image, alpha, zeros_mask, beta, gamma)

            cv.putText(mask_img, 'Regist', (20,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv.putText(mask_img, 'Delete', (20,110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            if RegistName is not None:
                cv.putText(mask_img, RegistName, (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            if DeleteName is not None:
                cv.putText(mask_img, DeleteName, (100,110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            return mask_img
        except:
            print('异常')

#鼠标回调函数
def detectMouseClick(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        points.append([x, y])
        

# def detectMouseClick
        
if __name__ == '__main__':
    # Step 2: load image
    # img1 = cv.imread("3.png")
    # img2 = cv.imread("4.png")
    # img1 = cv.resize(img1, (640, 480))
    # img2 = cv.resize(img2, (640, 480))
    points = [[0,0]]

    regist_flag = False
    delete_flag = False
    
    cv.namedWindow('img')
    cv.setMouseCallback('img',detectMouseClick)

    capture = cv.VideoCapture(0)
    face = Face()
    string = ''

    cv.namedWindow('User Guide')
    zeros = np.ones((320,700,3), dtype=np.uint8)*255
    cv.putText(zeros, 'User Guide', (20,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '1.Regist or Delete SHOULD Click the button and input', (20,70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  the corresponding username.', (20,90), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '2.Repeatedly Click the Regist or Delete button BUT ', (20,120), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  without press the ENTER key on your keyboard will', (20,140), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  be ignored, the system will only remember the ', (20,160), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  FIRST CLICK.', (20,180), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '3.The system will record the identities in a database', (20,210), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  for future use.', (20,230), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '4.You can press the ESC key on your keyboard to ', (20,260), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '  exit the system.', (20,280), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(zeros, '5.Close this prompt to continue using.', (20,310), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow('User Guide', zeros)

    i = 0
    while True:
        i+=1
        ret, frame = capture.read()
        if i%10!=0:
            continue
        
        img1 = cv.resize(frame, (640, 480))

        x,y = points[-1]
        if x>20 and x<180 and y>20 and y<60 and delete_flag == False:
            regist_flag = True
        elif x>20 and x<180 and y>80 and y<120 and regist_flag == False:
            delete_flag = True

        if regist_flag == True:
            img = face.put_mask(img1, RegistName=string)
        elif delete_flag == True:
            img = face.put_mask(img1, DeleteName=string)
        else:
            img = face.put_mask(img1)

        c = cv.waitKey(1)

        if c==27: #退出
            face.update()#每个程序都要有
            print('Exit!')
            break

        if c>0 and (c>=ord('a') and c<=ord('z')) or (c>=ord('A') and c<=ord('Z')):
            string += chr(c)
            # print(string)
        elif c==13: #回车
            print(string)
            if regist_flag:
                face.regist(img, string)
                regist_flag = False
                points.pop(-1)
            elif delete_flag:
                face.delete(string)
                delete_flag = False
                points.pop(-1)
            string = ''
        elif regist_flag==False and delete_flag==False:
            match_user = face.recognize(img)
            if match_user != None:
                print(match_user)

        cv.imshow('img',img)
    cv.destroyAllWindows()
    
    # face.regist(img1, 'A')
    # face.regist(img1, 'A')
    # face.recognize(img2)
    # # face.update()
    # # face.recognize(img2)
    # face.delete('A')
    

    # print(score)
    

    # score >= cosine similarity threshold, matched
    # score <= l2 distance threshold, matched
