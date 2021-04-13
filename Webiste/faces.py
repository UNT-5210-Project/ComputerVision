from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(sess)
face_classifier=cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

pred_list = ['anger','disgust','fear','happiness','neutrality','sadness','surpise']
class DetectEmotion(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        ret,frame=self.cap.read()
        labels=[]
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)

            # img = cv2.imread('Test_Images/happiness/'+str(j)+'.png')
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # imgs = cv2.resize(gray, (28,28), interpolation = cv2.INTER_CUBIC)
            # imgs = np.asarray(imgs, dtype='float32')
            # imgs = imgs.reshape(-1,28,28,1)
            # imgs = imgs/255.0
            # pred = model.predict_classes(imgs)
            # print(pred_list[pred[0]])
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(28,28),interpolation=cv2.INTER_AREA)
            # img = np.asarray(roi_gray, dtype='float32')
            # img.reshape(-1,28,28,1)
            # img = img/255.0 
            
            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                global sess
                global graph
                with graph.as_default():
                    set_session(sess)
                    # model = load_model('static/model_test.h5')
                    # pred = model.predict_classes(img)
                    print("Performing prediction")
                    # label=pred_list[pred[0]]
                    label=pred_list[1]
                    label_position=(x,y)
                    print(label)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes())
