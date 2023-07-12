import cv2
import numpy as np

path='./project_2_cavity_number/'
res_path = './results/'
report_img_path = '../report/img/'

#cap
cap_minDist=120
cap_param1=200
cap_param2=60
cap_minRadius=220
cap_maxRadius=230

#background
bg_minDist=120
bg_param1=200
bg_param2=60
bg_minRadius=250
bg_maxRadius=0

dx1=1.
dx2=1.
dy1=.2
dy2=.6

warp_polar_flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
kernel = np.ones((7,11),np.uint8)
morph_op = cv2.MORPH_OPEN
