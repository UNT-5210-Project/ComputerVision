from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(sess)
face_classifier=cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
model = load_model('static/best_model.h5')

# class_labels=['Angry','Happy','Neutral','Sad','Surprise']
# class_labels = ['anger','disgust','fear','happiness','neutrality','sadness','surpise']
class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
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
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(28,28),interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                global sess
                global graph
                with graph.as_default():
                    set_session(sess)
                    print("Predicting....")
                    preds=model.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    perentage_value = "{:.2f}".format(preds[preds.argmax()] * 100)
                    percentage_str = perentage_value + '%'
                    label_position=(x,y)
                    display_value = label + ": " + percentage_str
                    print(display_value)
                cv2.putText(frame,display_value,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes())
