from dataclasses import dataclass
import cv2
import numpy as np
import face_recognition
import os
import xlrd
import pandas as pd
from datetime import datetime

data = {'Name':['alia bhatt', 'emma stone', 'samantha','srk','tamannah bhatia'],
        'Age':[27, 24, 32,21,22],
        'CGPA':['70%', '80%', '23%','66%','73%'],
        'Total fee':[10000,20000,30000,25000,65000]}
df = pd.DataFrame(data)
    
path = "../student face data"
images = []
classNames = []
mylist = os.listdir(path)
for cls in mylist:
    curImg=cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode =  face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def studentDetail(name):
    print(df.loc[df['Name'] == name])

def markStudentid(name):
    with open('../student history.csv','r+') as f:
        myDataList = f.readlines()
        nameList=[]
        for line in myDataList:
            entry= line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
    
encodeListknown = findEncodings(images)
print('Encoding complete')

student = cv2.VideoCapture(0)

while True:
    sucess, img = student.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    curface = face_recognition.face_locations(imgS)
    encodecurf =  face_recognition.face_encodings(imgS,curface)

    for enocodeFace,FaceLoc in zip(encodecurf,curface):
        matches = face_recognition.compare_faces(encodeListknown,enocodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,enocodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255,),0)
        studentDetail(classNames[matchIndex])
        markStudentid(classNames[matchIndex])



    
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
