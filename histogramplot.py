# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:23:20 2018

@author: Sreeja
"""

import cv2
import os
import sys
from natsort import natsorted
import numpy as np
from matplotlib import pyplot as plt
import shutil
#arg1:video file name
#arg2:location where frames are stored
vidname=sys.argv[1]
dir1 = sys.argv[2]
dir2=sys.argv[3]
if os.path.exists(dir1):
    shutil.rmtree(dir1)
os.makedirs(dir1)
vidcap = cv2.VideoCapture(vidname)
success,image = vidcap.read()
count = 0
success = True
while success:
  if (count % 25 == 0):
      cv2.imwrite(os.path.abspath(dir1)+"/frame%d.jpg" % count, image)     
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1

image_folder=dir1
images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith(".jpg")]
hc=[]
print ("total number of frames in original video=",len(images))
histo=[]


for i in range(len(images)-1): 																																																																																																																																																																																																																																																																																																																																																																																																																																																																												
	img1=cv2.imread(os.path.join(os.path.abspath(dir1),images[i]))
	img2=cv2.imread(os.path.join(os.path.abspath(dir1),images[i+1]))	
	hist1 = cv2.calcHist([img1],[0, 1, 2],None,[8, 8, 8],[0, 256, 0, 256, 0, 256])
	hist2 = cv2.calcHist([img2],[0, 1, 2],None,[8, 8, 8],[0, 256, 0, 256, 0, 256])
	histo.append(np.asarray(hist2).flatten())
	hc.append(cv2.compareHist(hist1,hist2,4))
	
histo=np.asarray(histo)
diff=[]
for n in range(len(hc)-1):
	diff.append(abs(hc[n+1]-hc[n]))
mean1=np.mean(diff)
t=mean1
print ('threshold =',t)
print (diff)
plt.figure(figsize=(6,4))
plt.xlabel("Frame numbers")
plt.ylabel("Histogram difference")
plt.plot(hc,linewidth=3.0)
x=range(0,len(images))
y=[t]*len(x)

plt.plot(x,y,linewidth=3.0,color='r',linestyle='dashed')
plt.legend(['histogram difference','threshold'])
plt.show()
print("famecount=",count)
'''

s=[0]

for m in range(len(images)-2):
	if(diff[m]>t):
		s.append(m)

r=[]
for i in s:
	if i not in r:
		r.append(i)
print (r)
print ("Number of keyframes without clustering",len(r))
if os.path.exists(dir2):
    shutil.rmtree(dir2)
os.makedirs(dir2)
for k in range(len(r)):
	j=r[k]
	img3=cv2.imread(os.path.join(os.path.abspath(dir1),images[j]))
	cv2.imwrite(os.path.abspath(dir2)+"/frame%d.jpg" % j,img3) 


video_name = 'summ.avi'
images = [img for img in natsorted(os.listdir(dir2)) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(os.path.abspath(dir2),images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(os.path.join(os.path.abspath(dir2),video_name), cv2.VideoWriter_fourcc('M','J','P','G'), 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(os.path.abspath(dir2),image)))'''
  
