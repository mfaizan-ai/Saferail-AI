class Config:
    # RTSP URLS
    RTSP_URL_OPTICAL = 'rtsp://admin:hik*1245@10.19.30.100:554/Streaming/Channels/101'
    RTSP_URL_THERMAL = 'rtsp://admin:hik*1245@10.19.30.100:554/Streaming/Channels/201'
    
    # RTSP_URL_OPTICAL = 'videos/infer_visible2.mp4'
    # RTSP_URL_THERMAL = 'videos/infer_thermal2.mp4'
    # Display options
    WINDOW_NAME_OPTICAL = "Optical Stream"
    WINDOW_NAME_THERMAL = "Thermal Stream"
    
    # Queue settings
    MAX_QUEUE_SIZE = 2
    
    RTSP_SOURCE = True
    
    # Weights files
    DETECTION_WEIGHTS = 'detection_weights/engine_files/yolov5s_rrlv4_b2.engine'
    SEGMENTATION_WEIGHTS = 'segmentation_weights/engine_files/yolov5n-seg-300E-rrlv1-b2-best.engine'
    
    CONF = 0.4
    
    SAVE = False
    
    SEG_PLUGIN_LIBRARY = "detection_segmentation/JetsonYolov5/yolov5/build/libmyplugins.so"
    DET_PLUGIN_LIBRARY = "JetsonYolov5/yolov5/buildv1/libmyplugins.so"
    
    
    