import os
import cv2
import numpy as np
import pandas as pd
import time
import math
import src.config as config

def create_df(path, print=False):
    """
    Create dataframe of images and their features or labels
    """
    data=[]

    for img in os.listdir(path):
        cols = {'image':[]}
        cols['image']=img
        data.append(cols)

    df = pd.DataFrame(data)
   
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

def select_cc(img, numLabels, labels, stats, centroids, show=False, verbose=False):
    """
    This function takes as input a list of connected components and their stats,
    then for each cc draws its bounding box and compute its form factor.
    At the end returns the cc with the max form factor.
    """
    # list of blob form factors
    ffs = []
    ccs = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # the first component refers to background
        if i == 0:
            if verbose:
                print(f"[INFO] - examining component {i + 1}/{numLabels} (background)")
                print('Skipping background component...')
            continue
        # the others represent actual connected components
        else:
            if verbose:
                print(f"[INFO] - examining component {i + 1}/{numLabels}")
        
        ccs = draw_cc(ccs, centroids[i], stats[i])

        # construct a mask for the current connected component by
        # finding a pixels in the labels array that have the current
        # connected component ID
        componentMask = (labels == i).astype("uint8") * 255
        
        # masked_ccs = cv2.cvtColor(componentMask.copy(), cv2.COLOR_GRAY2BGR)
        # masked_ccs = draw_cc(masked_ccs, centroids[i], stats[i])

        # Compute form factor
        form_factor = compute_ff(componentMask, stats[i])

        if verbose:
            print(f'Blob {i}\'s form factor: {np.round(form_factor,2)}')

        ffs.append(form_factor)

        if show:
            # show our output image and connected component mask
            cv2.imshow("Output", ccs)
            # cv2.imshow("Connected Component", masked_ccs)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    return np.argmin(ffs)+1 # +1 to account for skipped background

def draw_cc(img, centroids, stats):
    """
    Draws an image aligned bounding box and the barycenter of a connected component.
    """
    # extract the connected component statistics and centroid for the current label
    x = stats[cv2.CC_STAT_LEFT]
    y = stats[cv2.CC_STAT_TOP]
    w = stats[cv2.CC_STAT_WIDTH]
    h = stats[cv2.CC_STAT_HEIGHT]
    (cX, cY) = centroids

    # draw a rectangle and its center as cc bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.circle(img, (int(cX), int(cY)), 1, (0, 0, 255), -1)

    return img

def compute_ff(cc, stats):
    """
    Computes form factor as 4*pi*A/P^2.
    """    
    # Compute form factor
    contours = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    p = len(contours[0])
    form_factor = p**2 / 4*np.pi*stats[cv2.CC_STAT_AREA]
    # form_factor = 4*np.pi*stats[cv2.CC_STAT_AREA] / p**2

    return form_factor

def link_cc_to_center(img, center, centroid):
    """
    Draws a line from cap's center to tab's barycenter.
    """    
    a, b = center
    (cX, cY) = centroid

    line_to_cc = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.circle(line_to_cc, (int(cX), int(cY)), 1, (0, 0, 255), -1)
    cv2.line(line_to_cc, (a, b), (int(cX), int(cY)), (0,0,255), 1, cv2.LINE_AA)

    return line_to_cc

def get_vertical_aligned_rot_matrix(rotation_center, reference_point):
    """
    Computes the rotation matrix M that aligns the line through tab and cap's center to the vertical axis.
    To correctly rotate the image, all the four possible quadrants must be handled.
    """ 
    cX, cY = reference_point
    a, b = rotation_center
    dy = cY - b
    dx = cX - a
    alpha = np.abs(np.arctan((cY - b)/(cX - a)))

    if dx > 0 and dy > 0: # I quad in a standard Cartesian system
        # print('I quad')
        theta = math.degrees(np.pi/2+alpha)
    elif dx < 0 and dy < 0: # II quad in a standard Cartesian system
        # print('II quad')
        theta = math.degrees(np.pi*3/2+alpha)

    elif dx < 0 and dy > 0: # III quad in a standard Cartesian system
        # print('III quad')
        theta = math.degrees(np.pi*3/2-alpha)

    elif dx > 0 and dy < 0: # IV quad in a standard Cartesian system
        # print('IV quad')
        theta = math.degrees(np.pi/2-alpha)

    # rotate our img by theta degrees around the img
    M = cv2.getRotationMatrix2D((a, b), theta, 1.0)

    return M, np.round(theta,0)

def rotate_point(point, M):
    """
    Apply a rotation matrix M to a point.
    """
    x, y = point
    p = np.array([x, y ,1])
    rp=np.dot(M, p)
    return rp

def crop_cavity_number(img, center, centroid):
    """
    Crop the cavity number. The crop dimensions depend on user defined parameters dy1, dy2, dx1, dx2.
    """
    a,b = center
    rcY = int(centroid[1])
    y1 = rcY + int((b-rcY)*config.dy1)
    y2 = rcY + int((b-rcY)*config.dy2)
    x1 = int(a-(y2-y1)*config.dx1)
    x2 = int(a+(y2-y1)*config.dx2)

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1 ,x2 ,y2)

def polar_rectification(img, center, radius):
    """
    Performs polar warping given its center and radius.
    """
    rectified_img = cv2.warpPolar(img, (-1,-1), center, maxRadius=radius, flags=config.warp_polar_flags)
    rectified_img = cv2.rotate(rectified_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rectified_img


def find_enclosing_rectangle(img, show=False, save=False):
    """
    This function takes an image as input and draws the maximal enclosed rectangle in it by applying a heuristic search.
    Credits to: https://stackoverflow.com/questions/21410449/how-do-i-crop-to-largest-interior-bounding-box-in-opencv/21479072#21479072
    """
    gray = img.copy()

    # Create a mask by selecting the non-zero values of the picture
    ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

    # Select the contour
    contours = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[0] if len(contours) == 2 else contours[1]

    # Get all the points of the contour
    contour = cont[0].reshape(len(cont[0]),2)

    # Assumption: rectangles with at least two points on the contour gives a 'good enough' result.
    # Get all possible rectangles based on this hypothesis.
    rect = []

    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2-y1)*abs(x2-x1)
            rect.append(((x1,y1), (x2,y2), area))

    # The first rect of all_rect has the biggest area, so it's the best solution if he fits in the picture
    all_rect = sorted(rect, key = lambda x : x[2], reverse = True)

    # Only rectangle with borders not falling in the black area are kept
    if all_rect:
        
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
        
        # Check if the rectangle is a good solution
        while not best_rect_found and index_rect < nb_rect:
            
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            
            valid_rect = True
            
            # Search a black area in the perimeter of the rectangle (vertical borders)
            x = min(x1, x2)
            while x <max(x1,x2)+1 and valid_rect:
                if mask[y1,x] == 0 or mask[y2,x] == 0:
                    # If a black pixel is found, that means a part of the rectangle is black
                    # so this rectangle is discarded.
                    valid_rect = False
                x+=1
            
            y = min(y1, y2)
            while y <max(y1,y2)+1 and valid_rect:
                if mask[y,x1] == 0 or mask[y,x2] == 0:
                    valid_rect = False
                y+=1
                
            if valid_rect:
                best_rect_found = True
            
            index_rect+=1
            
        if best_rect_found:
            rectangle_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).copy()
            if show:
                cv2.rectangle(rectangle_image, (x1,y1), (x2,y2), (0,0,255), 1)
                cv2.imshow("Best rectangle found",rectangle_image)
                if save:
                    cv2.imwrite(f"{config.report_img_path}rectified_img_encl_rect.jpg", rectangle_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            y1, y2, x1, x2 = min(y1, y2), max(y1, y2), min(x1,x2), max(x1,x2)
            
            # Finally, we crop the picture and store it
            result = img[y1:y2, x1:x2]

        else:
            print("No rectangle fitting into the area")
        
    else:
        print("No rectangle found")

    return result, ((x1, y1), (x2, y2))

def process_all(df, image_collection, smooth=False, verbose=False, show_progress=False, show_in_func=False, check_rotation=False):
    """
    Function that represents the end-to-end pipeline for project 2.
    """
    coords_col = []
    proc_time = []
    if not os.path.exists(config.res_path):
        os.mkdir(config.res_path)
    
    for i in df['image']:
        print(f'Processing {i}')
        img = image_collection[i]
        start = time.time()
        
        ## Cap detection
        if smooth:
            smoothed = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=1.5)
            cap = single_detect_circles(smoothed, config.cap_minDist, config.cap_param1, config.cap_param2, config.cap_minRadius, config.cap_maxRadius, print_result=False, show=show_in_func)
        else:
            cap = single_detect_circles(img, config.cap_minDist, config.cap_param1, config.cap_param2, config.cap_minRadius, config.cap_maxRadius, print_result=False, show=show_in_func)
        cap_a, cap_b ,cap_r = cap

        if verbose:
            print(f'Cap center = ({cap_a},{cap_b})')
            print(f'Cap radius = {cap_r}px')

        ## Background detection
        bg = single_detect_circles(img, config.bg_minDist, config.bg_param1, config.bg_param2, config.bg_minRadius, config.bg_maxRadius, print_result=False, show=show_in_func)
        bg_a, bg_b ,bg_r = bg

        if verbose:
            print(f'Background center = ({bg_a},{bg_b})')
            print(f'Background radius = {bg_r}px')
        
        # Image binarization
        binary = cv2.threshold(img, 0, 255,	cv2.THRESH_OTSU)[1]
        if show_progress:
            cv2.imshow('Binary image',binary)
            cv2.waitKey(0)
        
        masked = binary.copy()

        # Draw a circular mask of the size of the cap
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cap_mask = cv2.circle(mask, (cap_a,cap_b), cap_r, 255, -1)
        cap_masked = cv2.bitwise_and(masked, masked, mask=~cap_mask)
        
        if show_progress:
            cv2.imshow('Masked cap',cap_masked)
            cv2.waitKey(0)

        # Draw a circular mask of the size of the external industrial apparatus 
        bg_mask = cv2.circle(mask, (bg_a,bg_b), bg_r, 255, -1)
        bg_masked = cv2.bitwise_and(cap_masked, cap_masked, mask=bg_mask)

        if show_progress:
            cv2.imshow('Masked environment',bg_masked)
            cv2.waitKey(0)
        
        # Apply morphology open with 7x11 structuring element to cancel out cap's border residues   
        morph = cv2.morphologyEx(bg_masked, config.morph_op, config.kernel)

        if show_progress:
            cv2.imshow('morph',morph)
            cv2.waitKey(0)

        # Search for connected components and their features
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(morph, 4, cv2.CV_32S)
        
        # Select connected componente with highest form factor in order to discard spurious cc generated from borders
        selected_cc = select_cc(img, numLabels, labels, stats, centroids, show=show_in_func, verbose=verbose) # cup's tab
        line_to_cc = link_cc_to_center(img, [cap_a, cap_b], centroids[selected_cc])
        
        if show_progress:
            cv2.imshow("line_to_cc", line_to_cc)
            cv2.waitKey(0)
        
        # Compute rotation matrix from anywhere to vertical aligned cap's tab
        M, theta = get_vertical_aligned_rot_matrix([cap_a, cap_b], centroids[selected_cc])
        if verbose:
            print(f'Rotation of {theta} degrees')
        
        (h, w) = img.shape[:2]

        # Rotate the img by theta degrees around the cap center
        rotated = cv2.warpAffine(line_to_cc, M, (w, h))
        img_rotated = cv2.warpAffine(img.copy(), M, (w, h))

        if show_progress:
            cv2.imshow("Rotated by theta Degrees", rotated)
            cv2.imshow("Original image rotated by theta Degrees", img_rotated)
            cv2.waitKey(0)
        
        if check_rotation:
            # Use this cell to check rotation
            check = rotated.copy()
            cv2.line(check, (cap_a, cap_b), (cap_a, 0), (0,0,255), 1, cv2.LINE_AA)

            cv2.imshow("Rotated by theta Degrees", check)
            cv2.waitKey(0)

        # Compute rotated centroid coords and use them to crop the cavity number
        c = centroids[selected_cc]
        rc = rotate_point(c, M)
        crop, vertices = crop_cavity_number(img_rotated, [cap_a, cap_b], rc)
        if show_progress:
            cv2.imshow("crop", crop)
            cv2.waitKey(0)

        # Polar rectification: center is cap's center, radius is cap's radius
        x1, y1 = vertices[:2]
        center = (int(cap_a-x1), int(cap_b-y1))
        rectified_img = polar_rectification(crop, center, cap_r)
        
        if show_progress:
            cv2.imshow("rectified_img", rectified_img)
            cv2.waitKey(0)
        
        # Find enclosing rectangle and crop rectified image
        rectified_crop, coords = find_enclosing_rectangle(rectified_img, show=False)
        proc_time.append(time.time()-start)
        coords_col.append(coords)
        
        if rectified_crop is not None:
            cv2.imwrite(f'{config.res_path}{i}', rectified_crop)

    new_df = pd.DataFrame(data={'cavity_num_bbox':coords_col, 'proc_time':proc_time}, columns=['cavity_num_bbox', 'proc_time'])
    df = pd.concat([df, new_df], axis=1)
    return df