
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[3]:


img = cv2.imread('4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original',img)
cv2.imshow('gray',gray)

# In[4]:


faces = face_cascade.detectMultiScale(gray, 1.3, 5)


# In[ ]:


for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# In[ ]:


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

