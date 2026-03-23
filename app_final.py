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

# Send detected frames over a socket
def detect( source, 
            detection_model,  
            segmentation_model,
            host='127.0.0.1', 
            port=5001, 
            socket_status=False,
            save=False):
    """
    Detect object in a source i.e. RTSP stream or video file and send the detected frames over a socket if enabled. 
    
    Parameters
    ----------
    source: str
        Path to the  video (optical) or thermal.
    
    detection_model: YoloTRT object
        Object detection model (YOLO).
    host: str, optional
        The server IP address, default is localhost.
    port: int, optional
        The server port, default is 5001.
    """
    # Display properties
    font_scale = 0.5
    thickness = 1
    font_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if socket_status:
        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        print('Listening on host')
    
    # Initialize video captures
    cap = cv2.VideoCapture(source)
    
    # Video writing properties
    durations = []
    if save:
        frame_height, frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or *'mp4v' for mp4 format
        out = cv2.VideoWriter('videos/output.mp4',fourcc, fps, (frame_height, frame_width))

    durations = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
        
            
            if not ret:
                break
            
            tic = time.time()
            # Resize 
            frame= imutils.resize(frame, width=640)
            
            # frame = np.stack([frame_visible, frame_thermal], axis=0)
            frame_s = frame.copy()
            # black half image frame_s
            frame_s[:, :frame_s.shape[1]//2, :] = 0
            result_image, use_time = segmentation_model.infer([frame, frame_s])
            # Object detection
            detections, _ = detection_model.Inference([result_image[0], result_image[1]])

            
            # Draw detections (bounding boxes, labels) on the frame
            if not socket_status:
                duration = (time.time() - tic)
                durations.append(duration)
                fps_info = f'Current FPS: {(1/duration): .3f}, Average FPS: {((1/np.mean(durations))): .3f}'
                (text_width, text_height), baseline = cv2.getTextSize(fps_info, font, font_scale, thickness)
                text_pos = (10, text_height + 10)
                cv2.rectangle(frame, (text_pos[0], text_pos[1] - text_height - baseline), (text_pos[0] + text_width, text_pos[1] + baseline), bg_color, cv2.FILLED)
                cv2.putText(frame, fps_info, text_pos, font, font_scale, font_color, thickness, cv2.LINE_AA)                
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # show frame_s
                cv2.imshow('detectoin1', cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB))
                
                # Save the inference video
                if save and frame is not None and frame.size > 0:
                    out.write(frame)
                
            # Serialize the frame and send it over the socket
            if socket_status:
                # Convert frame to bytes
                data = pickle.dumps(frame)
                size = len(data)
                client_socket.sendall(struct.pack(">L", size) + data)
                print(f"Sent {size} bytes")
                
            # Destroy all windows
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if save:
            out.release()
        if socket_status:
            client_socket.close()       
    
        cap.release()
        cv2.destroyAllWindows()
        segmentation_model.destroy()
        

def detect_updated( source: list, 
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


    # Open two video sources
    cap1 = cv2.VideoCapture(source[0])  # Optical
    cap2 = cv2.VideoCapture(source[1])  # Thermal

    durations = []

    # Setup video writer
    if save:
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('videos/output.mp4', fourcc, fps, (width, height))

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            # time.sleep(0.1)

            if not ret1 or not ret2:
                print("[INFO] End of one or both videos.")
                break

            tic = time.time()

            # Resize both frames to match expected model input size
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

            # Save output
            if save:
                out.write(frame1)
                out.write(frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[INFO] Releasing resources...")
        if save:
            out.release()
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        segmentation_model.destroy()


if __name__ == "__main__":
    
    # Command Line Arguments
    parser = argparse.ArgumentParser(description="Get command line args.")
    parser.add_argument("--source1", type=str, default="video/optical.mp4", help="Path to the optical video or an rtsp stream")
    parser.add_argument('--source2', type=str, default="video/thermal.mp4", help="Path to the thermal video or an rtsp stream")
    parser.add_argument("--rtsp_source", action="store_true", help="using rtsp source or not.")
    parser.add_argument("--det_weights", type=str, default="engine_files/yolov5m_rrlv2.engine", help="Engine file to be deployed on Jetson.")
    parser.add_argument("--seg_weights", type=str, default="detection_segmentation/JetsonYolov5/yolov5/build_old/build/yolov5n-seg-300E-rrlv1-best.engine", help="Engine file for segmentation model.")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold for object detection.")
    parser.add_argument("--yolo_ver", type=str, default="v5", help="YOLO version.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--save", action="store_true", help="Save the output video.")
    args = parser.parse_args()
    
    # Track Segmentation configs
    SEG_PLUGIN_LIBRARY = "detection_segmentation/JetsonYolov5/yolov5/build/libmyplugins.so"

    ctypes.CDLL(SEG_PLUGIN_LIBRARY)
    
    SEG_CATAGORIES = ['track']

    # Create an instance of the YoLov5TRT class
    yolov5_segmentation_wrapper = YoLov5TRT(args.seg_weights)
    
    
    # YOLO definition
    yolo_detection_model = YoloTRT(library="JetsonYolov5/yolov5/buildv1/libmyplugins.so", engine= args.det_weights, conf=args.conf, yolo_ver= args.yolo_ver)
     
    # Run Object Detection
    detect_updated(
        source=[args.source1, args.source2],
        detection_model=yolo_detection_model,
        segmentation_model = yolov5_segmentation_wrapper,
        rtsp_source = args.rtsp_source,
        save=args.save)
