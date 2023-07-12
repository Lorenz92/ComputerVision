path='./project_1_liner_inspection/'
res_path = './results/'
report_img_path = '../report/img/'

avg_brightness = 76 # ceil of mean+3sigma of avg brightness
canny_lower = 0
canny_upper = 30
defect_t = 10000

#cap
cap_minDist=20
cap_param1=200
cap_param2=60
cap_minRadius=0
cap_maxRadius=0 

#liner
liner_minDist=100
liner_param1=50
liner_param2=30
liner_minRadius=180
liner_maxRadius=200