import cv2
import numpy as np
import face_recognition 
import os

pathname = 'ImagesAttendance'
imageslist = []
imagename = []
mylist = os.listdir(pathname)

for x in mylist:
    newImg = face_recognition.load_image_file(f'{pathname}/{x}')
    imageslist.append(newImg)
    imagename.append(os.path.splitext(x)[0])
#print(imagename)

def getEncodings(imageslist):
    listEncode = []
    for i in imageslist:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = cv2.resize(i, (0,0), None, 0.25,0.25)
        encode = face_recognition.face_encodings(i)[0]
        # d = face_recognition.face_locations(i)[0]
        # cv2.rectangle(i,(d[3],d[0]),(d[1],d[2]),(255,255,0),2)
        # cv2.imshow('e',i)
        # cv2.waitKey(0)
        listEncode.append(encode)
    return listEncode

encodeListKnown = getEncodings(imageslist)
print(len(encodeListKnown))

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgshrunk = cv2.resize(img,(0,0),None,0.25,0.25)
    imgshrunk = cv2.cvtColor(imgshrunk, cv2.COLOR_BGR2RGB)

    faceinFrame = face_recognition.face_locations(imgshrunk)[0]
    encodeinFrame = face_recognition.face_encodings(imgshrunk,faceinFrame)[0]

    for encodeFace, faceLoc in zip(encodeinFrame, faceinFrame):
        match = face_recognition.compare_faces(encodeListKnown, encodeFace)
        facedis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(facedis)
        print(facedis)
        
        if match[matchIndex]:
            name = imagename[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img,name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


