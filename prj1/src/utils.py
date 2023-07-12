import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import operator
import time
import src.config as config

def create_df(path, print=False):
    """
    Create dataframe of images and their features or labels
    """

    data=[]

    for img in os.listdir(path):
        cols = {'image':[], 'class':[], 'defect':[]}
        cols['image']=img
        cols['defect']='-'

        if img.startswith('d_'):
            cols['class']='defective'
        else:
            cols['class']='good'
        data.append(cols)

    df = pd.DataFrame(data)
    df.loc[df['image'] == 'd_31.bmp','defect']='missing_liner'
    df.loc[df['image'].isin(['d_17.bmp', 'd_18.bmp', 'd_19.bmp', 'd_20.bmp']),'defect']='incomplete_liner'
    if print:
        print(df)
    return df

def read_img_gs(path):
    """
    Read greyscale images from path
    """
    image_collection = {}

    for img in os.listdir(path):
        image = cv2.imread(path+img, cv2.COLOR_BGR2GRAY)
        image_collection[f'{img}'] = image
    
    return image_collection

def smooth(bgr_img, filter, show=False):
    """
    Applies median or bilateral filter of given size
    """
    gs = bgr_img.copy()
    if filter == 'bilateral':
        smooth = cv2.bilateralFilter(gs, 3, 1,1)
    else:
        smooth = cv2.medianBlur(gs, 5)
    
    if show:
        cv2.imshow('Binarized image',smooth)
        cv2.waitKey(0)
    return smooth

def single_detect_circles(image, minDist, param1, param2, minRadius, maxRadius, print_result=False, show=False, save_img_as=False):
    """
    HT to detect a single circle.
    """
    img=image.copy()
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if print_result:
        print(f'circles={circles}')

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,0,255),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    if show:
        cv2.imshow('detected circles',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_img_as:
        cv2.imwrite(f'{config.report_img_path}{save_img_as}',cimg)
    
    return circles.flatten()

def print_hist(img, mask=None, avg_bri=-1):
    """
    Print grey level histogram and average brightness if passed
    """    
    hist = cv2.calcHist([img],[0],mask,[256],[0,256])
    plt.figure()
    plt.stem(hist, use_line_collection=True)

    if avg_bri != -1:
        plt.axvline(avg_bri, color='k', linestyle='dashed', linewidth=1, label='image_avg_bri')
        plt.axvline(config.avg_brightness, color='r', linestyle='-', linewidth=1, label='threshold')
        plt.legend()

    # Add title and axis names
    plt.title('Color histogram')
    plt.xlabel('Gray levels')
    plt.ylabel('Counts')
    plt.show()

    return hist

def compute_avg_brightness(img, mask=None):
    """
    Compute average brightness of the input image
    """
    hist = cv2.calcHist([img],[0],mask,[256],[0,256])
    gray_levels = np.arange(256)
    average_brightness = np.dot(hist.flatten(),gray_levels)/np.sum(hist.flatten())
    return average_brightness

def line_length(line):
    """
    Compute length of the input line that as the form [[x1,y1,x2,y2]]
    """
    length = np.sqrt((line[0][2] -line[0][0])**2 + (line[0][3] -line[0][1])**2)
    return length

def check_defect_circular_region(img, lower, upper, circular_region, show=False):
    """
    This function smooths the input image with a median + Gaussian filter and then applies Canny's edge detector
    to find liner contours and defects
    """
    # Extracts center and radius
    a, b, r = circular_region

    # Smooth input image
    image = cv2.medianBlur(img, 3)
    image = cv2.GaussianBlur(image, (11, 11), 1.5)

    # Draw a circular mask of radius r-5
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (a,b), r-5, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Search for edges with Canny
    edged = cv2.Canny(masked, lower, upper, L2gradient=True)

    # Draw a circular mask of radius r-10
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (a,b), r-10, 255, -1)
    masked2 = cv2.bitwise_and(edged, edged, mask=mask)

    if show:
        cv2.imshow('Edges',edged)
        cv2.imshow('Only edges',masked2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edged.copy(), masked2.copy()

def draw_lines(img_bgr, linesP):
    """
    For each line passed as argument draws the line on the input image
    """
    img_lines = img_bgr.copy()
    for line in linesP:
        l = line[0]
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0,0,255), 1, cv2.LINE_AA)
    return img_lines


def draw_rect(img_bgr, cnt):
    """
    Draw the minimum enclosing rectangle of the passed contour on the input image
    """
    img_rect = img_bgr.copy()
    minArea = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(minArea)
    box = np.int0(box)
    cv2.drawContours(img_rect, [box], 0, (0, 0, 255), 1)
    return img_rect, box

def baseline_contours_finder(img_gray):
    """
    This function takes as input an image representing edges (from Canny for example) 
    and returns the minimum enclosing bounding box based on contours extrema.
    """
    img_cnt = img_gray.copy()
    
    # Find contours in input image
    contours = cv2.findContours(img_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Take the longest contour
    dict = {i: len(contours[i]) for i, c in enumerate(contours) }
    c = max(dict.items(), key=operator.itemgetter(1))[0]
    
    # Find extrema points
    cnt = np.vstack(contours[c]).squeeze()
    leftmost = cnt[cnt[:,0].argmin()]
    rightmost = cnt[cnt[:,0].argmax()]
    # topmost = cnt[cnt[:,1].argmin()]
    # bottommost = cnt[cnt[:,1].argmax()]
    extrema = [np.vstack([leftmost, rightmost]).reshape((1,4), order='C')]
    cnt = enclosing_rect(extrema)

    return cnt #np.vstack(contours[c]).squeeze()

def slht_contours(img_grey):
    """
    This function takes as input an image representing edges (from Canny for example) 
    runs a Hough transform to detect lines, takes the longest and
    returns the line minimum enclosing bounding box.
    """
    edged_img = img_grey.copy()
    
    # HT lines
    linesP = cv2.HoughLinesP(edged_img, 1, np.pi / 180, 60, minLineLength=10, maxLineGap=600)
    
    # Take the longest line
    arr = np.array(list(map(line_length, linesP)))
    index = np.argmax(arr)

    # Draw detected lines
    edged_img_bgr = cv2.cvtColor(edged_img, cv2.COLOR_GRAY2BGR)
    lines_img = draw_lines(edged_img_bgr, [linesP[index]])

    # Compute enclosing rectangle 
    contours = enclosing_rect([linesP[index]])
    
    return contours, lines_img

def mlht_contours(img_grey):
    """
    This function takes as input an image representing edges (from Canny for example) 
    runs a Hough transform to detect lines and
    returns the lines minimum enclosing bounding box.
    """
    edged_img = img_grey.copy()
    
    # HT lines
    linesP = cv2.HoughLinesP(edged_img, 1, np.pi / 180, 60, minLineLength=10, maxLineGap=600)

    # Draw detected lines
    edged_img_bgr = cv2.cvtColor(edged_img, cv2.COLOR_GRAY2BGR)
    lines_img = draw_lines(edged_img_bgr, linesP)

    # Compute enclosing rectangle 
    contours = enclosing_rect(linesP)
    
    return contours, lines_img

def enclosing_rect(linesP):
    """
    This function takes an array of lines as input, 
    computes the four vertices of the enclosing rectangle based on line orientation and returns them
    """

    points = []
    for index in range(len(linesP)):
        l = linesP[index][0]
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]

        # Drawing enclosing rectangle
        a = np.arctan((y2-y1)/(x2-x1))
        # Use ipot to compute dx and dy as shift around the points, in order to get the enclosing rectangle
        ipot=10
        dx = np.int0(ipot * np.cos(np.pi/2 +a))
        dy = np.int0(ipot * np.sin(np.pi/2 +a))
        maxd=np.max([np.abs(dx),np.abs(dy)])
        dx = maxd * dx/np.abs(dx)
        dy = maxd * dy/np.abs(dy)

        # Compute rectangle vertices as shifts of (x1,y1), (x2,y2)
        x_1_1, y_1_1, x_1_2, y_1_2 = np.int0(x1+dx), np.int0(y1+dy), np.int0(x1-dx), np.int0(y1-dy)
        x_2_1, y_2_1, x_2_2, y_2_2 = np.int0(x2+dx), np.int0(y2+dy), np.int0(x2-dx), np.int0(y2-dy)
        points.append([[x_1_1, y_1_1], [x_1_2, y_1_2], [x_2_1, y_2_1], [x_2_2, y_2_2]])

    # Minimum enclosing rectangle
    v = np.array(points)
    cnt = np.array(v.reshape(v.shape[0]*v.shape[1],2))
    return cnt

def process_all(df, image_collection, defect_hl_method, smooth=False, verbose=False, print_plot=False, show_in_func=False):
    """
    Function that represents the end-to-end pipeline for project 1.
    """

    detected_defect = []
    coords = []
    proc_time=[]
    list_cap_a = []
    list_cap_b = []
    list_cap_r = []
    list_avg_bri = []
    
    if not os.path.exists(config.res_path):
        os.mkdir(config.res_path)
    
    for i in df['image']:
        print(f'Processing {i}')
        rect_coords = []
        img = image_collection[i]
        start = time.time()

        ### Classification phase ###
        if smooth:
            img = smooth(img, 'median')

        ## Red cap detection
        cap = single_detect_circles(img, config.cap_minDist, config.cap_param1, config.cap_param2, config.cap_minRadius, config.cap_maxRadius, print_result=False, show=show_in_func)

        cap_a, cap_b ,cap_r = cap
        if verbose:
            print(f'Cap center = ({cap_a},{cap_b})')
            print(f'Cap radius = {cap_r}px')

        # Create mask of the size of the cap
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (cap_a, cap_b), cap_r, 255, -1)

        # Compute average brightness on masked histogram
        avg_bri = compute_avg_brightness(img, mask)

        if print_plot:
            _ = print_hist(img, mask, avg_bri)
            
        if verbose:
            print(f'Average brightness = {np.round(avg_bri,0)}')
        
        if avg_bri > config.avg_brightness:
            end = time.time()
            print(f'No liner detected in {i}')
            detected_defect.append('missing liner')
            coords.append(rect_coords)
            proc_time.append(end-start)
            list_cap_a.append(cap_a)
            list_cap_b.append(cap_b)
            list_cap_r.append(cap_r)
            list_avg_bri.append(np.round(avg_bri,0))
            continue
        
        else:

            ## Liner detection
            liner = single_detect_circles(img, config.liner_minDist, config.liner_param1, config.liner_param2, config.liner_minRadius, config.liner_maxRadius, print_result=False, show=show_in_func)
            liner_a, liner_b ,liner_r = liner

            if verbose:
                print(f'Liner center = ({liner_a},{liner_b})')
                print(f'Liner radius = {liner_r}px')
            
            edged_full, edge_only = check_defect_circular_region(img, config.canny_lower, config.canny_upper, [liner_a, liner_b, liner_r], show=show_in_func)
            
            # Check if there are edges (= defects)
            if np.sum(edge_only.flatten()) <= config.defect_t: # np.sum(edge_only.flatten()) number of white pixels
                end = time.time()
                detected_defect.append('-')

            ### Defect localization ###
            else:
                result = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
                if defect_hl_method == 'baseline':

                    # Baseline
                    cnt = baseline_contours_finder(edge_only)   
                
                elif defect_hl_method == 'slht':

                    # Hough Transform single line detection
                    cnt, lines_img = slht_contours(edged_full)
                    
                    if show_in_func:
                        cv2.imshow('Detected lines', lines_img)
                        cv2.waitKey(0)

                elif defect_hl_method == 'mlht':

                    # Hough Transform multiple lines detection
                    cnt, lines_img = mlht_contours(edged_full)
                    
                    if show_in_func:
                        cv2.imshow('Detected lines', lines_img)
                        cv2.waitKey(0)

                result, rect_coords = draw_rect(result, cnt)
                end = time.time()
                cv2.imwrite(f'{config.res_path}{i}', result)
                # i = i.split('.')[0] # for report
                # cv2.imwrite(f'{config.report_img_path}{i}.jpg', result)
                detected_defect.append('incomplete liner')
        coords.append(rect_coords)
        proc_time.append(end-start)
        list_cap_a.append(cap_a)
        list_cap_b.append(cap_b)
        list_cap_r.append(cap_r)
        list_avg_bri.append(np.round(avg_bri,0))
    list_avg_bri_no_max = sorted(list_avg_bri)[:-1]
    u_avg_bri, std_avg_bri = np.mean(list_avg_bri_no_max), np.std(list_avg_bri_no_max)
    print(f'Mean average brightness: {u_avg_bri}')
    print(f'Avg brightness std dev.: {np.round(std_avg_bri,2)}')
    new_df = pd.DataFrame(data={'detected_defect':detected_defect, 'cap_a':list_cap_a, 'cap_b':list_cap_b ,'cap_r':list_cap_r, 'avg_brightness':list_avg_bri, 'bbox':coords, 'proc_time':proc_time}, columns=['detected_defect', 'cap_a', 'cap_b', 'cap_r', 'avg_brightness', 'bbox', 'proc_time'])
    df = pd.concat([df, new_df], axis=1)
    return df