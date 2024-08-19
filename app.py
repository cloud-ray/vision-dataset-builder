# app.py
import os
import cv2
import supervision as sv
from ultralytics import YOLOWorld
from collections import defaultdict
import signal
import sys
from config.configs import *


# Initialize dictionaries
detected_objects = defaultdict(list)
consecutive_detections = defaultdict(int)

def signal_handler(sig, frame):
    print('Signal received, exiting program.')
    sys.exit(0)

# Create necessary directories for screenshots and labels
def create_directories():
    directories = [ORIGINAL_IMAGE_DIR, ORIGINAL_LABEL_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Load a pretrained YOLOv8s-worldv2 model
def initialize_model():
    model = YOLOWorld(MODEL_PATH)
    classes = MODEL_CLASSES
    model.set_classes(classes)
    return model

# Initialize Video Capture
def initialize_video_capture(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        exit()
    return cap

# Motion Detection
def detect_motion(frame, back_sub):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    fg_mask = back_sub.apply(blurred_frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    motion_pixels = cv2.countNonZero(thresh)
    return motion_pixels, fg_mask

# Perform Object Detection
def perform_object_detection(model, frame, TRACKER_CONFIG_PATH):
    results = model.track(
        source=frame,
        persist=True,
        tracker=TRACKER_CONFIG_PATH,
        conf=0.5,
        iou=0.5,
        classes=None,
        verbose=True
    )
    return results

def save_screenshot(ORIGINAL_IMAGE_DIR, class_name, class_id, confidence, track_id, screenshot_count_for_object, scaled_frame):
    screenshot_path = os.path.join(ORIGINAL_IMAGE_DIR, f"{class_name}_{class_id}_{confidence:.6}_{track_id}_SSC_{screenshot_count_for_object}.jpg")
    cv2.imwrite(screenshot_path, scaled_frame)
    print(f"Screenshot saved: {screenshot_path}")


def save_bbox_coordinates(ORIGINAL_LABEL_DIR, class_name, class_id, confidence, track_id, screenshot_count_for_object, bbox, original_width, original_height):
    x_tl, y_tl, x_br, y_br = map(int, bbox)
    bbox_label_path = os.path.join(ORIGINAL_LABEL_DIR, f"{class_name}_{class_id}_{confidence:.6}_{track_id}_SSC_{screenshot_count_for_object}.txt")
    with open(bbox_label_path, 'w') as f:
        f.write(f"{class_id} {original_width} {original_height} {x_tl} {y_tl} {x_br} {y_br}")
    print(f"BBOX coordinates saved: {bbox_label_path}")

def main():

    model = initialize_model()
    cap = initialize_video_capture(VIDEO_SOURCE)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=BG_HISTORY, varThreshold=BG_THRESHOLD, detectShadows=BG_SHADOWS)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    create_directories()

    # Initialize a counter for the screenshots
    object_screenshot_count = {}
    consecutive_no_tracker_count = 0
    object_detection_active = False

    # Retrieve original video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print(f"Original FPS: {fps}, Frame shape: {w}x{h}")

    while True:
        for _ in range(FRAMES_TO_SKIP):
            ret = cap.grab()  # Grab frames but don't decode them
            if not ret:
                print("Error: Unable to grab frame")
                break
        
        # Read the next frame to process
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break

        # Resize the frame by 50%
        scaled_frame = cv2.resize(frame, (0, 0), fx=SCALE_FRAME_WIDTH, fy=SCALE_FRAME_HEIGHT)

        # Count non-zero pixels in the thresholded image
        motion_pixels, fg_mask = detect_motion(scaled_frame, back_sub)

        # Check if object detection is currently active
        if object_detection_active:
            results = perform_object_detection(model, scaled_frame, TRACKER_CONFIG_PATH)
            # print(f"Speed: Preprocess: {results[0].speed['preprocess']:.2f} ms, Inference: {results[0].speed['inference']:.2f} ms, Postprocess: {results[0].speed['postprocess']:.2f} ms")
            orig_shape = results[0].orig_shape
            original_height, original_width = orig_shape

            # Iterate through each result if results is a list
            for result in results:

                detections = sv.Detections.from_ultralytics(result)
                class_names = detections.data['class_name'].tolist()
                confidences = detections.confidence.tolist()
                class_ids = detections.class_id.tolist()
                xyxy = detections.xyxy.tolist()

                if detections.tracker_id is not None:
                    track_ids = detections.tracker_id.tolist()
                else:
                    track_ids = [None] * len(class_names)

                # Update the detected_objects dictionary
                for class_name, confidence, track_id, class_id, bbox in zip(class_names, confidences, track_ids, class_ids, xyxy):
                    # print(f"Class Name: {class_name}, Class ID: {class_id}, Confidence: {confidence:.3f}, Track ID: {track_id}")
                    key = (class_name, track_id)

                    # Update consecutive detection count
                    if confidence >= CONFIDENCE_LEVEL: 
                        consecutive_detections[key] = consecutive_detections.get(key, 0) + 1

                        # Print the number of consecutive frames detected
                        print(f"Class Name: {class_name}, Class ID: {class_id}, Confidence: {confidence:.3f}, Track ID: {track_id} detected for {consecutive_detections[key]} consecutive frames.")

                        # Check if the object has been detected with high confidence for 10 consecutive frames
                        if consecutive_detections[key] == CONSECUTIVE_FRAMES: 

                            # Check if the maximum number of screenshots has been reached for this object
                            if key not in object_screenshot_count or object_screenshot_count[key] < MAX_SCREENSHOTS:
                                screenshot_count_for_object = object_screenshot_count.get(key, 0) + 1
                                object_screenshot_count[key] = screenshot_count_for_object

                                save_screenshot(ORIGINAL_IMAGE_DIR, class_name, class_id, confidence, track_id, screenshot_count_for_object, scaled_frame)
                                save_bbox_coordinates(ORIGINAL_LABEL_DIR, class_name, class_id, confidence, track_id, screenshot_count_for_object, bbox, original_width, original_height)

                                # Get the current screenshot count for this object
                                i = object_screenshot_count.get(key, 0)

                                # Print the current screenshot count
                                print(f"Class Name: {class_name}, Class ID: {class_id}, Confidence: {confidence:.3f}, Track ID: {track_id} reached {i} of {MAX_SCREENSHOTS} screenshots")
                                
                                # Reset consecutive detection count to avoid repeated screenshots
                                consecutive_detections[key] = 0

                            # Check if any object has reached the maximum number of screenshots
                            if all(count >= MAX_SCREENSHOTS for count in object_screenshot_count.values()):
                                print("Maximum number of screenshots reached for all objects.")
                                for key in consecutive_detections:
                                    consecutive_detections[key] = 0  # Reset consecutive detection count for all objects
                                object_detection_active = False
                                break

                    else:
                        # Reset the count if confidence drops
                        consecutive_detections[key] = 0

                if class_names:  # if any object is being detected
                    consecutive_no_tracker_count = 0
                    # print("Object is being detected")
                else:
                    consecutive_no_tracker_count += 1
                    print(f"Consecutive frames with no detection: {consecutive_no_tracker_count}")

                if consecutive_no_tracker_count >= 30:
                    consecutive_no_tracker_count = 0
                    print("Closing window due to 30 consecutive frames with no tracker_id")

                    object_detection_active = False
                    break

        else:
            if motion_pixels > MOTION_THRESHOLD + HYSTERESIS_DEADBAND:
                print("Motion Detected")
                for key in consecutive_detections:
                    consecutive_detections[key] = 0  # Reset consecutive detection count for all objects
                object_screenshot_count = {}  # Reset object screenshot count dictionary
                object_detection_active = True

            elif motion_pixels < MOTION_THRESHOLD - HYSTERESIS_DEADBAND:
                # Switch to motion detection
                print("Motion Not Detected")

        # Display the resized frame and the foreground mask
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
