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
import json
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

import torch
from torchvision.transforms import ToTensor, Resize, ToPILImage
from torch import Tensor
# from JetsonYolov5.yoloDet import YoloTRT
from JetsonYolov5.yolo_det import YoloTRT
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, ycbcr_to_rgb

import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from detection_segmentation.JetsonYolov5.app_segmentation import YoLov5TRT


# Converts time to milliseconds
def get_ms(tic, toc):
    """get time"""
    return (toc - tic) * 1000


# Function to encode and send a frame and metadata
def send_frame_and_metadata(conn, frames, metadata):
    # Prepare frame sizes and encode frames
    frame_sizes = []
    encoded_frames = []

    for frame in frames:
        _, encoded_frame = cv2.imencode(".jpg", frame)
        frame_bytes = encoded_frame.tobytes()
        frame_sizes.append(len(frame_bytes))
        encoded_frames.append(frame_bytes)

    # Create metadata payload
    payload = {
        "metadata": metadata,
        "frame_sizes": frame_sizes,  # Include the size of each frame
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    payload_length = struct.pack(">I", len(payload_bytes))

    # Send metadata first (length + data)
    conn.sendall(payload_length)
    conn.sendall(payload_bytes)

    # Send each frame
    for frame_bytes in encoded_frames:
        conn.sendall(frame_bytes)



# Function to get frames from the video source
def get_frames(caps):
    """
    Get frames from the video source.
    
    Parameters
    ----------
    cap: cv2.VideoCapture
        Video capture object.
    
    Returns
    -------
    frame: np.ndarray
        The captured frame.
    """
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Failed to read frame from video source.")
            continue
        frames.append(frame)
    return frames

def detect(
    frames,
    detection_model,
    segmentation_model,
):
    
    # Object detection
    if len(frames) == 2:
        frame1 = imutils.resize(frames[0], width=640)
        frame2 = imutils.resize(frames[1], width=640)
        result_images, _, polygons, overlays = segmentation_model.infer([frame1, frame2])
        # result_image, use_time = segmentation_model.infer([frames])
        # detections, _ = detection_model.Inference(result_image[0])
        detections, _ = detection_model.Inference([result_images[0], result_images[1]], polygons, overlays, alpha=0.5)
        return detections,  [frame1, frame2]  
    else:
        raise ValueError("Expected exactly two frames for detection, got {}".format(len(frames)))
    
    
        
    
        
# Send detected frames over a socket
def stream(sources, 
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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[INFO] Listening on {host}:{port}")
        
        
        durations = []
        while True:
            # wait for the client to connect
            conn, addr = s.accept()
            print(f"[INFO] Connected to {addr}")
            
            # Open video source 
            caps = [cv2.VideoCapture(v) for v in sources]
            
            try:
                with conn:
                    while True:
                        
                        # get frames from each video source
                        
                        frames = get_frames(caps)
                        if not frames:
                            print("[INFO] No more frames. Exiting stream.")
                            break
                                        
                        # detections
                        tic = time.time()
                        detections, frames = detect(frames, detection_model, segmentation_model)
                                    
                                    
                        # Put FPS info on the frames
                        duration = (time.time() - tic)
                        durations.append(duration)
                        fps_info = f'Current FPS: {(1/duration): .3f}, Average FPS: {((1/np.mean(durations))): .3f}'
                        (text_width, text_height), baseline = cv2.getTextSize(fps_info, font, font_scale, thickness)
                        text_pos = (10, text_height + 10)
                        cv2.rectangle(frames[0], (text_pos[0], text_pos[1] - text_height - baseline), (text_pos[0] + text_width, text_pos[1] + baseline), bg_color, cv2.FILLED)
                        cv2.putText(frames[0], fps_info, text_pos, font, font_scale, font_color, thickness, cv2.LINE_AA) 
                        
                        cv2.rectangle(frames[1], (text_pos[0], text_pos[1] - text_height - baseline), (text_pos[0] + text_width, text_pos[1] + baseline), (255, 255, 255), cv2.FILLED)
                        cv2.putText(frames[1], fps_info, text_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA) 
                        
                                               
                        # add detections and meta data
                        # labels = ["person", "vehicle", "bike", "animal"]
                        metadata = {
                            "timestamp": cv2.getTickCount(),
                            "detections": None,
                        }
                        send_frame_and_metadata(conn, [frames[0]], metadata)
                        
                        
            except (ConnectionResetError, ConnectionAbortedError) as e:
                # Handle client disconnection
                print(f"Client disconnected: {e}")
            except Exception as e:
                print(f"Error: {e}")
            finally:
                # Release video captures
                for cap in caps:
                    cap.release()
            print(f"Connection with {addr} closed.")
            os._exit(0)
            
            

if __name__ == "__main__":
    
    # Command Line Arguments
    parser = argparse.ArgumentParser(description="Get command line args.")
    parser.add_argument("--source1", type=str, default="video/optical.mp4", help="Path to the optical video.")
    parser.add_argument("--source2", type=str, default="video/thermal.mp4", help="Path to the thermal video.")
    parser.add_argument("--det_weights", type=str, default="engine_files/yolov5m_rrlv2.engine", help="Engine file to be deployed on Jetson.")
    parser.add_argument("--seg_weights", type=str, default="detection_segmentation/JetsonYolov5/yolov5/build/yolov5n-seg-300E-rrlv1-b2-best.engine", help="Engine file for segmentation model.")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold for object detection.")
    parser.add_argument("--yolo_ver", type=str, default="v5", help="YOLO version.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--save", action="store_true", help="Save the output video.")
    args = parser.parse_args()
    
    HOST = '10.19.30.141'
    PORT = 9999
    socket_communication = True
    
    
    # Track Segmentation configs
    SEG_PLUGIN_LIBRARY = "detection_segmentation/JetsonYolov5/yolov5/build/libmyplugins.so"

    ctypes.CDLL(SEG_PLUGIN_LIBRARY)
    
    SEG_CATAGORIES = ['track']
    
    # Create an instance of the YoLov5TRT class
    yolov5_segmentation_wrapper = YoLov5TRT(args.seg_weights)

    
    # YOLO definition
    yolo_model = YoloTRT(library="JetsonYolov5/yolov5/build/libmyplugins.so", engine= args.det_weights, conf=args.conf, yolo_ver= args.yolo_ver)
    
    sources = [args.source1, args.source2]
    # Run Object Detection
    stream(
        sources=sources,
        detection_model=yolo_model,
        segmentation_model = yolov5_segmentation_wrapper,
        host= HOST, 
        port= PORT,
        socket_status=socket_communication,
        save=args.save)
