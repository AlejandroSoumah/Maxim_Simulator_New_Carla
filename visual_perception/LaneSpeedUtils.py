
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
save_image=True



def corners_unwarp(undist, nx, ny, mtx, dist):
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    #findchessboardcorners takes grayscale image, patternSize in this case it is 9x6, corners will be the output array of detected corners.
    # FLAGS => None, some valid options are CALIB_CB_FAST_CHECK to avoid looking for chessboard.
    img_size = (undist.shape[1], undist.shape[0])
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    if ret:
        # drawChessboardCorners will render  the detected chessboard corners.
        #Takes in undistrorted image 8bit, patternsize (9x6), corners(returned from findchessboardcorners call), ret(return val)
        cv2.drawChessboardCorners(undist, (nx,ny), corners, ret)
        if ret:
            #Coordinates of quadrangle vertices in the source image.
            src = np.float32([corners[0][0], corners[nx - 1][0], corners[-1][0], corners[-nx][0]])
            #Coordinates of quadrangle vertices in the destination image.
            dst = np.float32([[100, 100], [1150,100], [1150, 650], [100, 650]])
            #Calculate a perspective transform from four pairs of the corresponding points src & dst
            M = cv2.getPerspectiveTransform(src, dst)
            #Now using the transformation matrix (3x3) we can apply the warpPerspective on undistorted image
            # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
            warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
            
    return warped, M



def bounding_box(img, roi_points):
    mask = np.zeros_like(img)
    img_size = img.shape
    vertices = np.array(roi_points, dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
        #print(ignore_mask_color)
    else:
        ignore_mask_color = 255
        
    #The function fillPoly fills an area bounded by several polygonal contours
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    roi_image = cv2.bitwise_and(img, mask)
    return roi_image


# In[4]:


def image_unwarp(img, roi_points):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(roi_points)
    dst = np.float32([[0, 0], [640, 0], [640, 720], [0, 720]])
    # Given src and dst points, calculate the perspective transform matrix
    M  = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (640,720), flags=cv2.INTER_LINEAR)
    return warped


# In[5]:


def abs_sobel_thresh(img, orient, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # GaussianBlur filter will reduce or removes noise while keeping edges relatively sharp.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if (orient=='x'):
        sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary


# In[6]:


def xhl_thresh(img, x_thresh, h_thresh, l_thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    h_binary = np.zeros_like(h)
    h_binary[(h > h_thresh[0]) & (h <= h_thresh[1])] = 1
    l_binary = np.zeros_like(l)
    l_binary[(l > l_thresh[0]) & (l <= l_thresh[1])] = 1
    hl_binary = np.zeros_like(l)
    hl_binary[(h_binary == 1) | (l_binary == 1)] = 1
    
    sxbinary = abs_sobel_thresh(img, 'x', 5, x_thresh)
    xhl_binary = np.zeros_like(sxbinary)
    xhl_binary[(hl_binary == 1) & (sxbinary == 1)] = 1
    
    return xhl_binary


# In[7]:



def mag_thresh(img, sobel_kernel, mag_thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary

def dir_threshold(img, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(gradir)
    binary[(gradir >= thresh[0]) & (gradir <= thresh[1])] = 1
    return binary

def rgb_thresh(img, channel, thresh):
    channel = img[:, :, channel]
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel < thresh[1])] = 1
    return binary
def hls_thresh(img, channel_num, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,channel_num]
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary
def hsv_thresh(img, channel_num, thresh):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channel = hsv[:,:,channel_num]
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary
def color_threshold(img, channel='rgb', thresh=(220,255)):
    #Convert the image to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if channel is 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif channel is 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif channel is 'yuv':    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif channel is 'ycrcb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif channel is 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    elif channel is 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    
    img_ch1 = img[:,:,0]
    img_ch2 = img[:,:,1]
    img_ch3 = img[:,:,2]

    bin_ch1 = np.zeros_like(img_ch1)
    bin_ch2 = np.zeros_like(img_ch2)
    bin_ch3 = np.zeros_like(img_ch3)

    bin_ch1[(img_ch1 > thresh[0]) & (img_ch1 <= thresh[1])] = 1
    bin_ch2[(img_ch2 > thresh[0]) & (img_ch2 <= thresh[1])] = 1
    bin_ch3[(img_ch3 > thresh[0]) & (img_ch3 <= thresh[1])] = 1
    
    return bin_ch1, bin_ch2, bin_ch3
def combined_color(img):
     bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 = color_threshold(img, channel='rgb', thresh=(230,255))
     bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 = color_threshold(img, channel='hsv', thresh=(230,255))    
     bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 = color_threshold(img, channel='luv', thresh=(157,255))

     binary = np.zeros_like(bin_rgb_ch1)
    
     binary[(bin_rgb_ch1 == 1) | (bin_hsv_ch3 == 1) | (bin_luv_ch3 == 1) ] = 1
    
     return binary
# combine sobel x and value 
def xv_thresh(img, x_thresh, s_thresh):
    sxbinary = abs_sobel_thresh(img, 'x', 3, x_thresh)
    v_binary = hsv_thresh(img, 2, s_thresh)

    xv_binary = np.zeros_like(sxbinary)
    xv_binary[(sxbinary == 1) & (v_binary == 1)] = 1

    return xv_binary

#combine value and blue
def vb_thresh(img, v_thresh, b_thresh):    
    v_binary = hsv_thresh(img, 2, v_thresh)
    b_binary = rgb_thresh(img, 2, b_thresh)
    vb_binary = np.zeros_like(sxbinary)
    vb_binary[(b_binary == 1) | (v_binary == 1)] = 1
    return vb_binary

#combine Saturation & Value
def sv_thresh(img, s_thresh, v_thresh):
    v_binary = hsv_thresh(img, 2, v_thresh)
    s_binary = hsv_thresh(img, 1, s_thresh)
    
    sv_binary = np.zeros_like(s_binary)
    sv_binary[(s_binary == 1) | (v_binary == 1)] = 1
    
    return sv_binary

#combine value, sobel x filter, L value from HSL
def xvl_thresh(img, x_thresh, v_thresh, l_thresh):
    sxbinary = abs_sobel_thresh(img, 'x', 5, x_thresh)
    v_binary = hsv_thresh(img, 2, v_thresh)

    xv_binary = np.zeros_like(v_binary)
    xv_binary[(sxbinary == 1) & (v_binary == 1)] = 1
    
    l_binary = hls_thresh(img, 1, l_thresh)
    xvl_binary = np.zeros_like(xv_binary)
    xvl_binary[(xv_binary == 1) | (l_binary == 1)] = 1
    
    return xvl_binary

def hs_thresh(img, h_thresh, s_thresh):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hls[:,:,0]
    s = hls[:,:,1]
    h_binary = np.zeros_like(h)
    h_binary[(h > h_thresh[0]) & (h <= h_thresh[1])] = 1
    s_binary = np.zeros_like(s)
    s_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1
    hs_binary = np.zeros_like(s)
    hs_binary[(h_binary == 1) | (s_binary == 1)] = 1
    return hs_binary

def xhv_thresh(img, x_thresh, h_thresh, v_thresh):
    #tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, (9,9))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    v = hsv[:,:,2]
    h_binary = np.zeros_like(h)
    h_binary[(h > h_thresh[0]) & (h <= h_thresh[1])] = 1
    v_binary = np.zeros_like(v)
    v_binary[(v > v_thresh[0]) & (v <= v_thresh[1])] = 1
    hv_binary = np.zeros_like(v)
    hv_binary[(h_binary == 1) | (v_binary == 1)] = 1
    
     
    
    sxbinary = abs_sobel_thresh(img, 'x', 5, x_thresh)
    xhv_binary = np.zeros_like(sxbinary)
    xhv_binary[(hv_binary == 1) & (sxbinary == 1)] = 1
        
    
    return xhv_binary

def polynomial_fit(warped, left_indices, right_indices, left_fit, right_fit):
    
    if (len(left_indices) ==0 | len(right_indices) ==0 ):
        return left_fit, right_fit
    
    nonzeroy, nonzerox = warped.nonzero()
    
    leftx = nonzerox[left_indices]
    lefty = nonzeroy[left_indices]
    rightx = nonzerox[right_indices]
    righty = nonzeroy[right_indices]
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    if(((left_fit != []) & (right_fit != [])) & (left_fit[0] * right_fit[0] < 0.)):
        if(np.absolute(left_fit[1] - left_fit[1]) > 0.1):
            left_fit[0] = right_fit[0]
            left_fit[1] = right_fit[1]
            left_fit[2] = right_fit[2] - 450
        if(np.absolute(right_fit[1] - right_fit[1]) > 0.1):
            right_fit[0] = left_fit[0]
            right_fit[1] = left_fit[1]
            right_fit[2] = left_fit[2] + 450
            
    return left_fit, right_fit

#generate x and y values for plotting
def calculate_lane(left_fit, right_fit):
    left_poly = left_fit
    right_poly = right_fit
    ploty = np.linspace(0, 719, 720)
    left_fitx = left_poly[0] * ploty ** 2 + left_poly[1] * ploty + left_poly[2] 
    right_fitx = right_poly[0] * ploty ** 2 + right_poly[1] * ploty + right_poly[2]
    return left_fitx, right_fitx
def modified_windows(warped, left_fit, right_fit):
    left_poly = left_fit
    right_poly = right_fit
    ploty = np.linspace(0, 719, 720)
    nonzeroy, nonzerox = warped.nonzero()
    margin = 100
    
    #using previous fit data to predict the position of the indices of lane line
    left_pre = left_poly[0] * (nonzeroy**2)  + left_poly[1]*nonzeroy + left_poly[2]
    right_pre = right_poly[0] * (nonzeroy**2) + right_poly[1] * nonzeroy + right_poly[2]
        
    #find indices position in warp between prediction +/- margin
    left_indices = ((nonzerox > left_pre - margin) & (nonzerox < left_pre + margin))
    right_indices = ((nonzerox > right_pre - margin) & (nonzerox < right_pre + margin))
    
    return left_indices, right_indices

def tuning_draw(left_lane, right_lane, left_indices, right_indices, warped):
    left_fitx = left_lane
    right_fitx = right_lane
    
    ploty = np.linspace(0, 719, 720)
    nonzeroy, nonzerox = warped.nonzero()
    
    #create an image to draw 
    out_img = np.dstack((warped, warped, warped)) ** 255
    
    window = np.zeros_like(out_img)
    
    margin = 50
    
    # Generate a polygon to illustrate the search window area and recast the x and y points into cv2.fillPolly()
    
    left_window1 = np.array([np.transpose(np.vstack([left_lane - margin, ploty]))])
    left_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane + margin, ploty])))])
    left_pts = np.hstack((left_window1, left_window2))
    
    #Right Window
    right_window1 = np.array([np.transpose(np.vstack([right_lane - margin, ploty]))])
    right_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane + margin, ploty])))])
    right_pts = np.hstack((right_window1, right_window2))

    # color in left and right line pixels
    out_img[nonzeroy[left_indices], nonzerox[left_indices]] = [255, 0, 0]
    out_img[nonzeroy[right_indices], nonzerox[right_indices]] = [0, 0, 255]

    # draw the lane onto the warped blank image
    cv2.fillPoly(window, np.int_([left_pts]), (0, 255, 0))
    cv2.fillPoly(window, np.int_([right_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window, 0.3, 0)
    
    return result


# In[ ]:





# In[8]:



def sliding_window(warped,out):
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    nonzeroy, nonzerox = warped.nonzero()
    
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #the number of sliding windows
    nwindows = 9
    #set the height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    
    #current position to be updated for each window
    left_current = left_base
    right_current = right_base
    
    #set the width of the windows +/= margin
    margin = 70
    
    #set minimum number of pixels found to recenter window
    minpix = 50
    #create empty lists to recieve left and right lane pixes indices
    left_indices = []
    right_indices = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        #identify window boundaries in x and y (and right and left)
        win_low = warped.shape[0] - (window+1) * window_height  
        win_high = warped.shape[0] - window * window_height
        
        win_left_left = left_current - margin
        win_left_right = left_current + margin
        
        win_right_left = right_current - margin
        win_right_right = right_current + margin
        
        #draw the windows on visualization image (Rectangle drawing)
        #cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → None
        #https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.rectangle
        #Left Side
        cv2.rectangle(out, (win_left_left, win_low), (win_left_right, win_high), (0,255,0), 2)
        #Right Side
        cv2.rectangle(out, (win_right_left, win_low), (win_right_right, win_high), (0,255,0), 2)
        
        #Now we have the window lets identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_low) & (nonzeroy < win_high) & (nonzerox >= win_left_left) & (nonzerox < win_left_right)).nonzero()[0]
        
        #right indice
        good_right_inds = ((nonzeroy >= win_low)& (nonzeroy < win_high) & (nonzerox >= win_right_left) & (nonzerox < win_right_right)).nonzero()[0]
        
        #append these indices in the window to the list of all windows
        left_indices.append(good_left_inds)
        right_indices.append(good_right_inds)
        
        #if you find good indices > minpix pixels, recenter next window on thier mean poistion
        if len(good_left_inds) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        
    #concatenate all the arrays of indices
    left_indices = np.concatenate(left_indices)
    right_indices = np.concatenate(right_indices)
    
    leftx = nonzerox[left_indices]
    lefty = nonzeroy[left_indices]
    rightx = nonzerox[right_indices]
    righty = nonzeroy[right_indices]
    
    
    #draw left and right lane lines
    out[nonzeroy[left_indices], nonzerox[left_indices]] = [255, 0, 0]
    out[nonzeroy[right_indices], nonzerox[right_indices]] = [0, 0, 255]
       
    
    return left_indices, right_indices


# In[9]:


def combined_mono(warped):
    sx_binary = abs_sobel_thresh(warped, 'x', 9, (30, 100))
    sy_binary = abs_sobel_thresh(warped, 'y', 9, (30, 100))
    mag_binary = mag_thresh(warped, 9, (30, 100))
    dir_binary = dir_threshold(warped, 15, (0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((sx_binary == 1) & (sy_binary == 0)) | ((mag_binary == 0.6) & (dir_binary == 0.3))] = 1
    return combined


# In[10]:


### Build Lane finding pipeline.
class Line():
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.prev_left_poly = None
        self.prev_right_poly = None
        self.left_diff = None
        self.right_dif = None
        self.right_diffs = []
        self.left_diffs = []
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # threshold value
        self.THRESHOLD = 60
        self.MAX_WINDOW_SIZE = 20
        #
        self.last_n_fitted = np.zeros((self.MAX_WINDOW_SIZE, 2, 3), dtype=np.float64)
        self.window_size = 0
        self.last_fitted_idx = 0     
        self.best_fit = np.zeros((2, 3), dtype=np.float64)
        self.curve_radius = None 
        self.center_offset = None
        #frame count
        self.curr_frame_idx = 0
        #polynomial
        self.left_poly = None
        self.right_poly = None
        self.mid_poly = None
        self.left_bottom =  (80, 700)
        self.left_top = (560, 405)
        self.right_bottom = (1200, 700)
        self.right_top = (716, 405)
        
        self.roi_points = [[self.left_top, self.right_top, self.right_bottom, self.left_bottom]]
        self.undist  = None
    def camera_calibration(self,images, nx, ny):

        #create objp np.array with nx*ny items of type float32, 9x6 = 54 items of [0. 0. 0.]
        objp = np.zeros((ny*nx, 3), np.float32)

        #create a grid from [0,0]...[5,4]... [8,5]
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        #Arrays to store objpoints and imgpoints
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        # Step through the image list and search for chess board corners
        for fname in tqdm(images):
            img = cv2.imread(fname)
            img_size = (img.shape[1], img.shape[0])
            #since it is cv2.imread color format will be in BGR and not RGB
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            #If corners are found add object points and image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # once we have objpoints and imgpoints we can now calibrate using the cv2.calibrateCamera function
        # Which returns the camera matrix(mtx), distortion coefficients(dist), rotation(rvecs) and translation vectors(tvecs)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None, None)
        #img = mpimg.imread("camera_cal/calibration3.jpg")
        self.mtx = mtx
        self.dist = dist
   
    def validate(self, left_poly, right_poly, left_indices, right_indices):
        if left_poly is not None  and right_poly is not None:
            prev_left_poly, prev_right_poly = self.get_previous_fits()
            if prev_left_poly is not None and right_poly is not None:
                self.left_diff = np.linalg.norm(left_poly - prev_left_poly)
                self.right_diff = np.linalg.norm(right_poly - prev_right_poly)
                self.left_diffs.append(self.left_diff)
                self.right_diffs.append(self.right_diff)
                if (self.left_diff < self.THRESHOLD and self.right_diff < self.THRESHOLD) or (self.window_size == 0):
                    self.last_n_fitted[self.last_fitted_idx, 0, :] = left_poly
                    self.last_n_fitted[self.last_fitted_idx, 1, :] = right_poly
                    self.window_size = min(self.MAX_WINDOW_SIZE, self.window_size + 1)
                    self.last_fitted_idx = (self.last_fitted_idx + 1) % self.MAX_WINDOW_SIZE
            self.best_fit[0] = np.sum(self.last_n_fitted[:, 0, :], axis=0) / self.window_size
            self.best_fit[1] = np.sum(self.last_n_fitted[:, 1, :], axis=0) / self.window_size
        self.detected = True
    
    def get_previous_fits(self):
        return self.best_fit[0], self.best_fit[1]
    def image_rewarp(self,img):
        dst = np.float32(self.roi_points)
        src = np.float32([[0, 0], [640, 0], [640, 720], [0, 720]])
        M = cv2.getPerspectiveTransform(src, dst)
        rewarped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)
        return rewarped
    def draw_lane(self,warped, left_lane,right_lane):
        # create zero fille np.array
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        lane_draw = np.dstack((warp_zero, warp_zero, warp_zero))


        ploty = np.linspace(0, 719, 720)
        left_pts = np.array([np.transpose(np.vstack([left_lane, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_lane, ploty])))])
        pts = np.hstack((left_pts, right_pts))

        pts_trans_left  = left_pts.reshape(720,2)
        pts_trans_right = right_pts.reshape(720,2)
        pts_trans_left = np.array(pts_trans_left,dtype = np.float64)
        pts_trans_right = np.array(pts_trans_right,dtype = np.float64)

        pts_trans_right =np.flipud(pts_trans_right)
        pts_trans_mid = (pts_trans_left + pts_trans_right) /  2


        cv2.fillPoly(lane_draw, np.int_([pts]), (0, 220, 110))
        cv2.polylines(lane_draw, np.int_([left_pts]), isClosed=False,
                      color=(255, 255, 0), thickness=20)
        cv2.polylines(lane_draw, np.int_([right_pts]), isClosed=False,
                      color=(255, 255, 0), thickness= 20)


        cv2.polylines(lane_draw, np.int32([pts_trans_mid]), isClosed=False,
                      color=(255, 255, 255), thickness= 20)

        rewarped = self.image_rewarp(lane_draw)
        result = cv2.addWeighted(self.undist, 1, rewarped, 0.3, 0)

        return result,pts_trans_left,pts_trans_right,pts_trans_mid

    def find_lanes(self, img):
                undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
                self.undist = undist
                roi = bounding_box(undist, self.roi_points)
        
                warped = image_unwarp(roi, self.roi_points)
                #finding color space
                combined_binary = combined_mono(warped)
                out = np.dstack((combined_binary, combined_binary, combined_binary)) ** 255
                
                if (self.detected == False):
                        left_poly, right_poly = [], []
                        left_indices, right_indices = sliding_window(combined_binary,out)
                        left_poly, right_poly = polynomial_fit(combined_binary, left_indices, right_indices, left_poly, right_poly)
                        self.prev_left_poly, self.prev_right_poly = left_poly, right_poly
                        
                        self.detected = True
                else:
                         left_poly, right_poly = self.get_previous_fits()
                         left_indices, right_indices = modified_windows(combined_binary, left_poly, right_poly)
                         left_poly, right_poly = polynomial_fit(combined_binary, left_indices, right_indices, left_poly, right_poly)
                
                self.validate(left_poly, right_poly, left_indices, right_indices)
                left_lane, right_lane = calculate_lane(left_poly, right_poly)

                #Curvate and distance
                #curve_radii, center_offset = curvature_distance(left_lane, right_lane, left_poly, right_poly)
                
                #filled Line
                Lane_img,pts_left,pts_right,pts_mid = self.draw_lane(combined_binary, left_lane, right_lane)
                
                #self.curve_radius = curve_radii
                #self.center_offset = center_offset
                #result_img = self.add_measurements(undist, filled_lane, roi_points)
                self.curr_frame_idx += 1
                
               
                return Lane_img,pts_left,pts_right,pts_mid
                  