import cv2
import numpy as np
import pygame

pygame.init()
window=pygame.display.set_mode((1366,768))
cap=cv2.VideoCapture(0)

def f(x):
    return x

cv2.namedWindow('trackbars')

set=0
####PRESS 'S' ON KEYBOARD AFTER SETTING THE COLOUR######## 
cv2.createTrackbar('Lower H','trackbars',0,255,f)
cv2.createTrackbar('Higher H','trackbars',0,255,f)
cv2.createTrackbar('Lower S','trackbars',0,255,f)
cv2.createTrackbar('Higher S','trackbars',0,255,f)
cv2.createTrackbar('Lower V','trackbars',0,255,f)
cv2.createTrackbar('Higher V','trackbars',0,255,f)

while(cap.isOpened()):
    ret,frame1=cap.read()
    frame = cv2.resize(frame1,(1366, 768), interpolation = cv2.INTER_CUBIC)
    lh=cv2.getTrackbarPos('Lower H','trackbars')
    hh=cv2.getTrackbarPos('Higher H','trackbars')
    ls=cv2.getTrackbarPos('Lower S','trackbars')
    hs=cv2.getTrackbarPos('Higher S','trackbars')
    lv=cv2.getTrackbarPos('Lower V','trackbars')
    hv=cv2.getTrackbarPos('Higher V','trackbars')
    lower = np.array([lh,ls,lv])
    upper = np.array([hh,hs,hv])
    frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(frameHSV,lower,upper)
    res1=cv2.bitwise_and(frame,frame,mask=mask)
    res=cv2.flip(res1,1)
    cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1366,768)
    cv2.rectangle(res,(1250,10),(1350,60),(0,255,0),1)
    cv2.imshow('Result',res)  
    k=cv2.waitKey(1)

    if k==ord('q'):
        break
    if k==ord('s'):
        set=1
        
    if(set==1):
        ret,cnts,heirarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if(len(cnts)==0):
            continue
        else:
            maxcnt=max(cnts,key=cv2.contourArea)
            epsilon = 0.1*cv2.arcLength(maxcnt,True)
            approx = cv2.approxPolyDP(maxcnt,epsilon,True)
            M = cv2.moments(approx)
            if(M['m00']==0):
                continue
            else:      
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])                                  
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)                               
                pygame.draw.circle(window,(0,255,0),(cx,cy),5,5)
                pygame.display.flip()
                        
            cv2.drawContours(frame,approx,-1,(0,255,0),2)
            cv2.imshow('Track',frame)
                         
   
cv2.destroyAllWindows()
cap.release()
