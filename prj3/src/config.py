import cv2

path='./project_3_off-center_decoration/'
res_path = './results/'
report_img_path = '../report/img/'

# Dark Red cap color segmentation
red_lower_bound = (0, 210, 50)
red_upper_bound = (7, 250, 180)

# Grey decoration color segmentation
grey_lower_bound = (149, 0, 50)
grey_upper_bound = (179, 50, 180)

# Cap
cap_filter = 'median'
cap_struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

cap_minDist=5000
cap_param1=200
cap_param2=10
cap_minRadius=205
cap_maxRadius=0

#BKP
# cap_minDist=500
# cap_param1=200
# cap_param2=10
# cap_minRadius=195
# cap_maxRadius=0

# Decoration
decoration_filter = 'bilateral'
decoration_struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

# maxRadius is determined at runtime based on cap radius 
decoration_minDist=5000
decoration_param1=250
decoration_param2=1
decoration_minRadius=187

# OLD
# minDist=120
# param1=200
# param2=30
# minRadius=100
# maxRadius=r