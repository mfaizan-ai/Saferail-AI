'''
OBJECT DETECTOIN DEPLOYMENT ON JETSON ORIN NANO DEVELOPER KIT

Author: Muhammad Faizan
Date: 16 05 2024

Description:
This code implements a function to send detected frames over a socket for object detection.
'''

# Import necessary libraries
import cv2
import numpy as np
import socket
import struct
import pickle
import imutils
import time
from typing import Tuple
from pathlib import Path
import threading
import queue
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

import torch
from torchvision.transforms import ToTensor, Resize, ToPILImage
from torch import Tensor
# from JetsonYolov5.yoloDet import YoloTRT
from JetsonYolov5.yolo_det import YoloTRT
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, ycbcr_to_rgb
from track_roi import mask_to_points_per_batch, create_occupancy_region

import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from detection_segmentation.JetsonYolov5.app_segmentation import YoLov5TRT
from configs.config import Config

# thread configs
queue1 = queue.Queue(maxsize=1)
queue2 = queue.Queue(maxsize=1)
stop_event= threading.Event()


def rtsp_reader(name, url, frame_queue):
    while not stop_event.is_set():
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f'[{name}] Failed to open stream. Retrying...')
            cap.release()
            time.sleep(0.5)
            continue
        
        print(f'[{name}] Stream opened successfully.')
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print('f[{name}] Failed to read frame, reconnecting...')
                break
            
            
            if not frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
        cap.release()
        time.sleep(0.2)
    print(f'[{name}] Exiting reader thread.')
    
    

def find_roi_regions(per_batch_points):
    regions = []
    for per_image_pts in per_batch_points:
        occupied_region = create_occupancy_region(per_image_pts['mask1'], per_image_pts['mask2'])
        regions.append(occupied_region)
    return regions


# Converts time to milliseconds
def get_ms(tic, toc):
    """get time"""
    return (toc - tic) * 1000
        

def analyze_sources( source: list, 
            detection_model,  
            segmentation_model,
            rtsp_source = False,
            save=False):
    """
    Detect objects using two sources (optical + thermal) and optionally send the detected frames over a socket.

    Parameters
    ----------
    source : list[str]
        Paths to two video sources: [optical_video_path, thermal_video_path].

    detection_model : YoloTRT
        Object detection model wrapper.

    segmentation_model : YoLov5TRT
        Segmentation model wrapper.

    host : str
        IP address for socket streaming.

    port : int
        Port number for socket streaming.

    socket_status : bool
        If True, enables socket streaming.

    save : bool
        If True, saves output to disk.
    """
    assert len(source) == 2, "You must provide exactly two sources [optical, thermal]"

    # Fonts for FPS display
    font_scale = 0.5
    thickness = 1
    font_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    durations = []
    try:
        while not stop_event.is_set():
            try:
                if not queue2.empty() and not queue1.empty():
                    frame1 = queue1.get_nowait()
                    frame2 = queue2.get_nowait()
                    
                    # Resize the frames 
                    tic = time.time()
                    frame1 = imutils.resize(frame1, width=640, height=360 if rtsp_source else None)
                    frame2 = imutils.resize(frame2, width=640, height=360 if rtsp_source else None)
                    

                    # Run segmentation model with both sources
                    result_images, _, polygons, overlays = segmentation_model.infer([frame1, frame2])

                    # Run detection model on both segmentation outputs
                    detections, _ = detection_model.Inference([result_images[0], result_images[1]], polygons, overlays, 
                                                            alpha=0.5)

                    # Overlay FPS on original optical frame
                    duration = (time.time() - tic)
                    durations.append(duration)
                    fps_info = f'FPS: {(1/duration):.2f}, Avg FPS: {(1/np.mean(durations)):.2f}'
                    (text_w, text_h), base = cv2.getTextSize(fps_info, font, font_scale, thickness)
                    pos = (10, text_h + 10)
                    cv2.rectangle(frame1, (pos[0], pos[1] - text_h - base), (pos[0] + text_w, pos[1] + base), bg_color, cv2.FILLED)
                    cv2.putText(frame1, fps_info, pos, font, font_scale, font_color, thickness, cv2.LINE_AA)
                    
                    # thermal frame fps info display
                    cv2.rectangle(frame2, (pos[0], pos[1] - text_h - base), (pos[0] + text_w, pos[1] + base), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame2, fps_info, pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                
                    # show both frames 
                    cv2.imshow('Optical stream', frame1)
                    cv2.imshow('Thermal stream', frame2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
                
            except Exception as e:
                print(f'Display error: {e}')
                stop_event.set()
                break

    finally:
        print("[INFO] Releasing resources...")
        cv2.destroyAllWindows()
        segmentation_model.destroy()


if __name__ == "__main__":
    ctypes.CDLL(Config.SEG_PLUGIN_LIBRARY)
    
    # Create an instance of the YoLov5TRT class
    yolov5_segmentation_wrapper = YoLov5TRT(Config.SEGMENTATION_WEIGHTS)
    
    # YOLO Model
    yolo_detection_model = YoloTRT(library=Config.DET_PLUGIN_LIBRARY, engine= Config.DETECTION_WEIGHTS, conf=Config.CONF, yolo_ver= "v5")
    
    try:
        thread1 = threading.Thread(target=rtsp_reader, args=(Config.WINDOW_NAME_OPTICAL, Config.RTSP_URL_OPTICAL, queue1))
        thread2 = threading.Thread(target=rtsp_reader, args=(Config.WINDOW_NAME_THERMAL, Config.RTSP_URL_THERMAL, queue2))
        
        thread1.start()
        thread2.start()
        
         # Run video analytics (segmentatoin and detection)
        analyze_sources(
            source=[Config.RTSP_URL_OPTICAL, Config.RTSP_URL_THERMAL],
            detection_model=yolo_detection_model,
            segmentation_model = yolov5_segmentation_wrapper,
            rtsp_source = Config.RTSP_SOURCE,
            save= Config.SAVE)
    
        thread1.join()
        thread2.join()
        
        print('Clean shutdown completed.')
        
    except KeyboardInterrupt:
        print('Exiting due to keyboard interrupt')
        stop_event.set()
