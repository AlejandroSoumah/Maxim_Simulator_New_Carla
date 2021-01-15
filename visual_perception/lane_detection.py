import cv2
import matplotlib.pyplot as plt
from visual_perception.lane_detection_utils import process_image_rgb,get_linear_midpoint,pipeline,perspective_warp,get_curved_midpoints,sliding_window,get_curve,draw_lanes
from PIL import Image
def curved_lane_detection(rgb_input_img):
    dst = pipeline(rgb_input_img)
    dst = perspective_warp(dst, dst_size=(1280,720))
    out_img, curves, lanes, ploty = sliding_window(dst)
    curverad=get_curve(rgb_input_img, curves[0],curves[1])
    img_= draw_lanes(rgb_input_img, curves[0], curves[1])
    return img_
def linear_lane_detection(rgb_input_img):
    rgb_output_image,rgb_left_Lane,rgb_right_Lane = process_image_rgb(rgb_input_img)
    Mid_Img,Mid_Lane = get_linear_midpoint(rgb_input_img,rgb_left_Lane,rgb_right_Lane)
    return Mid_Img