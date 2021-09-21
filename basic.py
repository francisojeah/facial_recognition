import cv2
import numpy as np
import face_recognition 

imgAda = face_recognition.load_image_file('ImagesBasic/Adaglasses.jpg')
imgAda = cv2.resize(imgAda, (480,640))
imgAda = cv2.cvtColor(imgAda, cv2.COLOR_BGR2RGB)

imgAda2 = face_recognition.load_image_file('ImagesBasic/Adawig.jpg')
imgAda2 = cv2.resize(imgAda2, (480,640))
imgAda2 = cv2.cvtColor(imgAda2, cv2.COLOR_BGR2RGB)

imgAda1 = face_recognition.load_image_file('ImagesBasic/Adanormal.jpg')
imgAda1 = cv2.resize(imgAda1, (480,640))
imgAda1 = cv2.cvtColor(imgAda1, cv2.COLOR_BGR2RGB)

a =[imgAda,imgAda2]

faceLoc = face_recognition.face_locations(a)
encodeAda= face_recognition.face_encodings(a)[0]
print(faceLoc)
cv2.rectangle(imgAda,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,255,0),2)

#for x in faceLoc:
    #cv2.rectangle(imgAda,(x[3],x[0]),(x[1],x[2]),(0,0,255),2)

faceLocA = face_recognition.face_locations(imgAda1)[0]
encodeAda1 = face_recognition.face_encodings(imgAda1)[0]
cv2.rectangle(imgAda1,(faceLocA [3],faceLocA [0]),(faceLocA [1],faceLocA [2]),(0,255,0),2)

result = face_recognition.compare_faces([encodeAda],encodeAda1)
facedis = face_recognition.face_distance([encodeAda],encodeAda1)
print(facedis)
cv2.putText(imgAda1, f'{result} {round(facedis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,  1, (255,0,0),2)
cv2.imshow('Adanna', imgAda)
cv2.imshow('Ada', imgAda1)

cv2.waitKey(0)