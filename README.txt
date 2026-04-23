Regionally Optimized Video Object Detection and Tracking for UK Urban Traffic
Author: Zaamin Qadeer | W1906890 | University of Westminster

========================================
SETUP
========================================

Install dependencies:

    pip install torch torchvision ultralytics deep-sort-realtime opencv-python numpy

========================================
HOW TO RUN
========================================

1. Train the custom CNN (optional, weights already included)

    python customcnn.py

2. Run the tracking pipeline

    python customcnn_tracking_yolo.py

Make sure traffic_video.mp4 and custom_cnn_output/best_custom_model.pth
are in the same folder before running.

========================================
OUTPUT
========================================

Results are saved to output_hybrid/

    tracked_output_hybrid.mp4       annotated video with tracking IDs
    tracking_report_hybrid.txt      tracking stats report
    tracking_report_hybrid.json     machine readable report

========================================
RESULTS
========================================

    CNN Validation Accuracy     96.15%
    Detection mAP (IoU 0.5)     0.752
    Unique Vehicles Tracked     60
    Frames Processed            4,860