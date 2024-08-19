# Configuration variables

# Video source URL
VIDEO_SOURCE = 'http://192.168.4.24:8080/video'

# Model settings
MODEL_CLASSES = ['lamp', 'chair']

# Motion detection thresholds
MOTION_THRESHOLD = 1000
HYSTERESIS_DEADBAND = 200

# Background thresholds
BG_HISTORY = 300
BG_THRESHOLD = 100
BG_SHADOWS = True

# Confidence level and consecutive frames
CONFIDENCE_LEVEL = 0.70
CONSECUTIVE_FRAMES = 30

# Frame scaling factors
SCALE_FRAME_WIDTH = 0.5
SCALE_FRAME_HEIGHT = 0.5

# Video processing settings
FRAMES_TO_SKIP = 3
MAX_SCREENSHOTS = 2

# Directory paths
ORIGINAL_IMAGE_DIR = 'original_data/images/'
ORIGINAL_LABEL_DIR = 'original_data/labels/'

PROCESSED_IMAGE_DIR = 'processed_data/images/'
PROCESSED_LABEL_DIR = 'processed_data/labels/'
PROCESSED_DIR = 'processed_data/'

PLOT_IMAGE_DIR = 'plot_data/'

MODEL_PATH = 'model/yolov8s-worldv2.pt'
TRACKER_CONFIG_PATH = 'config/bytetrack.yaml'

# Roboflow upload configs
NUM_WORKERS = 10
DATASET_FORMAT = "yolov8"
PROJECT_LICENSE = "MIT"
PROJECT_TYPE = "object-detection"
