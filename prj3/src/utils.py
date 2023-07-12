import os
import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import src.config as config


def create_df(path, print=False):
    """
    Create dataframe of images and their features or labels
    """
    data=[]

    for img in os.listdir(path):
        cols = {'image':[], 'class':[]}
        cols['image']=img

        if img.startswith('d_'):
            cols['class']='defective'
        else:
            cols['class']='good'
        data.append(cols)

    df = pd.DataFrame(data)
    if print:
        print(df)
    return df

def read_img_rgb(path):
    image_collection = {}

    for img in os.listdir(path):
        image = cv2.imread(path+img)
        image_collection[f'{img}'] = image
    
    return image_collection

def read_img_hsv(path):
    """
    Read RGB images from path and convert them into HSV 
    """
    hsv_image_collection = {}

    for img in os.listdir(path):
        image = cv2.imread(path+img)
        hsv_image_collection[f'{img}'] = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    return hsv_image_collection

def plot_channel_hist(img):
    """
    Plot HSV histogram for each channel
    """
    (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

    for c in range(3):
        if c == 0:
            hist, bins = np.histogram(img[:,:,c].flatten(),180,[0,180])
        else:
            hist, bins = np.histogram(img[:,:,c].flatten(),256,[0,256])
        # plot each of the histograms
        axs[c].stem(hist, use_line_collection=True)
        # set the titles of each axes
        axs[c].set_title(f"Channel {c}")

    plt.show()

def color_segmentation(rgb_img, hsv_img, lb, ub):
    """
    Returns an image where pixels with value < lb or > ub are set to zero.
    In other words performs color segmentation by extracting regions which pixels value are between lb and ub.
    """    
    lower_bound = lb
    upper_bound = ub
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    red = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    return red

def smooth(bgr_img, filter, show=False):
    """
    Applies bilateral or median filters of fixed size based on user choice.
    """
    gs = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).copy()
    if filter == 'bilateral':
        smooth = cv2.bilateralFilter(gs, 3, 1,1)
    else:
        smooth = cv2.medianBlur(gs, 5)
    
    if show:
        cv2.imshow('Binarized image',smooth)
        cv2.waitKey(0)
    return smooth

# Process all images

def process_all(df, image_collection, hsv_image_collection, show_progess=False, verbose=False):
    """
    Function that represents the end-to-end pipeline for project 3.
    """
    list_cap_a = []
    list_cap_b = []
    list_cap_r = []
    list_decoration_a = []
    list_decoration_b = []
    list_decoration_r = []
    list_d = []
    proc_time = []
    if not os.path.exists(config.res_path):
        os.mkdir(config.res_path)
    for i in df['image']:
        print(f'Processing {i}')
        img = image_collection[i]
        hsv = hsv_image_collection[i]
        start = time.time()
        
        ### Cap ###
        # Color segmentation
        red = color_segmentation(img, hsv, config.red_lower_bound, config.red_upper_bound)
        
        # Smooth image
        cap_smooth = smooth(red, config.cap_filter)
        
        ## Red cap detection
        cap = single_detect_circles(cap_smooth, config.cap_minDist, config.cap_param1, config.cap_param2, config.cap_minRadius, config.cap_maxRadius, print_result=False)
        cap_a,cap_b,cap_r = cap
        
        ### Decoration ###
        # Let's focus only on the region contained in the detected cap circle
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        m = cv2.circle(m, (cap_a,cap_b), cap_r, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=m)

        # Color segmentation
        grey = color_segmentation(masked, hsv, config.grey_lower_bound, config.grey_upper_bound)
        
        # Smooth image
        decoration_smooth = smooth(grey, filter=config.decoration_filter)

        # Decoration circle detecion
        maxRadius = cap_r
        decoration = single_detect_circles(decoration_smooth, config.decoration_minDist, config.decoration_param1, config.decoration_param2, config.decoration_minRadius, maxRadius, print_result=False)
        decoration_a,decoration_b,decoration_r = decoration

        # Draw detected cap and decoration and save image
        cimg = draw_circles(img, [cap, decoration], show=show_progess)
        cv2.imwrite(f'{config.res_path}{i}', cimg)

        # Compute Euclidean distance between cap's center and decoration's center
        d = compute_distance(cap, decoration, verbose=verbose)
        
        proc_time.append(time.time()-start)
        list_cap_a.append(cap_a)
        list_cap_b.append(cap_b)
        list_cap_r.append(cap_r)
        list_decoration_a.append(decoration_a)
        list_decoration_b.append(decoration_b)
        list_decoration_r.append(decoration_r)
        list_d.append(d)

    # Update Dataframe
    new_df = pd.DataFrame(data={'cap_a':list_cap_a, 'cap_b':list_cap_b, 'cap_r':list_cap_r, 'deco_a':list_decoration_a, 'deco_b':list_decoration_b, 'deco_r':list_decoration_r, 'd':list_d, 'proc_time':proc_time}, columns=['cap_a','cap_b','cap_r','deco_a','deco_b','deco_r','d','proc_time'])
    df = pd.concat([df, new_df], axis=1)

    return df

def single_detect_circles(image, minDist, param1, param2, minRadius, maxRadius, print_result=False, show=False, save_img_as=False):
    """
    HT to detect a single circle.
    """
    img=image.copy()

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius) # cup mouth
    
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

def draw_circles(img, circles, show=False):
    """
    For each circle in circles draws it on the passed image.
    """
    cimg = img.copy()

    for i,c in enumerate(circles):
        a,b,r = c
        # draw the outer circle
        cv2.circle(cimg,(a,b),r,(i*255,0,255),2)

        # draw the center of the circle
        cv2.circle(cimg,(a,b),1,(i*255,0,255),1)
    if show:
        cv2.imshow('detected circles',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cimg


def compute_distance(cap, decoration, verbose):
    """
    Compute Euclidean distance between cap's center and decoration's center.
    """
    cap_center = np.float32(cap[:2])
    decoration_center = np.float32(decoration[:2])

    d = np.round(np.sqrt((cap_center[0]-decoration_center[0])**2 + (cap_center[1]-decoration_center[1])**2),2)
    if verbose:
        print(f'Distance between cap and decoration centers = {d}')
    return d