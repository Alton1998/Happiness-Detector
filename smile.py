import numpy as np
import cv2
import pickle
smile_cascade=cv2.CascadeClassifier('./data/haarcascade_smile.xml')
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
    smiles=smile_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=20)
    for(x,y,w,h) in smiles:
        color=(255,0,0)
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
