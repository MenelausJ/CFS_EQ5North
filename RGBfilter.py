# RBG filter for feature points
import sys
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


xleft = 2325
xright = 2460
ytop = 620
ybottom = 1700
msk = np.ones((ybottom-ytop, xright-xleft))
for x in range(msk.shape[1]):
	ymin = int(min(max(0, -565/31*x + 1080), msk.shape[0]))
	ymax = int(min(max(0, -565/31*x + 76303/31), msk.shape[0]))
	msk[:ymin, x] = 0
	msk[ymax:, x] = 0

dltkernel = np.ones([9,9])

frame_i = 891
with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/RawDisp.txt", "w") as f:
	while (frame_i <= 1605):
		frame = cv2.imread("/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame%d.png"%(frame_i))
		frame = frame[ytop:ybottom, xleft:xright]

		frame[frame[...,2]<120]=0
		frame[frame[...,1]<120]=0
		frame[frame[...,0]<120]=0
		frame[frame[...,2]>150]=0
		frame[frame[...,1]>150]=0
		frame[frame[...,0]>150]=0

		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = (frame*msk).astype('uint8')
		frame = gaussian_filter(frame,2)
		frame[frame<30]=0
		frame[frame>0]=255
		frame2 = cv2.dilate(frame, dltkernel)

		_1, contours, _2 = cv2.findContours(frame2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		fpts = []
		for i in range(len(contours)):
			cnt = contours[i]
			x, y, w, h = cv2.boundingRect(cnt)
			if min(w, h) > 8:
				# frame = cv2.rectangle(frame, (x+3, y+3), (x+w-3, y+h-3), (255, 255, 255), cv2.FILLED)
				fpts.append(np.array([x+4, y+4]))

		if len(fpts) != 6:
			print "RGB filter failed, modify params"
			print len(fpts)
			plt.imshow(frame2)
			plt.show()
			break

		fpts = sorted(fpts, key = lambda x:x[1])

		f.write(str(frame_i) + "\t")
		for i in range(len(fpts)):	
			f.write(str(fpts[i][0]) + "\t")
			f.write(str(fpts[i][1]) + "\t")
			
		f.write("\n")

		# cv2.imwrite("/Users/yujieli/Documents/CFS_Video_Analysis-master/ftpoints/feat%d.png"%(frame_i), frame)
		
		print frame_i
		frame_i += 1