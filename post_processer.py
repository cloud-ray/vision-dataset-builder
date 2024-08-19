# post_processer.py
from pybboxes import BoundingBox
import cv2
import os
import roboflow
import uuid
from config.configs import *
from dotenv import load_dotenv

load_dotenv()

# Create necessary directories for screenshots and labels
def create_directories():
    directories = [PROCESSED_IMAGE_DIR, PROCESSED_LABEL_DIR, PLOT_IMAGE_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_image(image_path):
    return cv2.imread(image_path)

def read_bbox_coordinates(bbox_txt_path):
    try:
        with open(bbox_txt_path, 'r') as file:
            return file.readline().strip().split()
    except FileNotFoundError:
        print(f"Text file not found: {bbox_txt_path}")
        return None

def extract_bbox_data(bbox_data):
    class_id = int(bbox_data[0])
    original_width = int(bbox_data[1])
    original_height = int(bbox_data[2])
    x_tl = int(bbox_data[3])
    y_tl = int(bbox_data[4])
    x_br = int(bbox_data[5])
    y_br = int(bbox_data[6])
    return class_id, original_width, original_height, x_tl, y_tl, x_br, y_br

def crop_image(image, original_width, original_height, x_tl, x_br):
    new_width = original_height
    image_height, image_width, _ = image.shape
    crop_x = (image_width - new_width) // 2

    if x_tl < crop_x:
        crop_x = x_tl
    elif x_br > crop_x + new_width:
        crop_x = x_br - new_width

    return image[:, crop_x:crop_x + new_width, :], crop_x

def save_image(image, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image)
    print(f'Image saved to {filepath}')

def draw_bounding_box(image, x_tl, y_tl, x_br, y_br):
    cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), (0, 255, 0), 2)
    return image

def save_yolo_labels(class_id, yolo_bbox, label_filepath):
    os.makedirs(os.path.dirname(label_filepath), exist_ok=True)
    with open(label_filepath, 'w') as f:
        f.write(f'{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}')
    print(f'YOLO results saved to {label_filepath}')

def convert_bbox_to_yolo(my_voc_box, new_width, new_height):
    voc_bbox = BoundingBox.from_voc(*my_voc_box, image_size=(new_width, new_height))
    return voc_bbox.to_yolo()

def confirm_yolo_conversion(yolo_bbox, my_voc_box):
    voc_bbox = BoundingBox.from_yolo(*yolo_bbox.raw_values, image_size=yolo_bbox.image_size).to_voc()
    if voc_bbox.raw_values == tuple(my_voc_box):
        print("YOLO conversion is successful!")
    else:
        print("YOLO conversion failed!")

def process_image(input_image_dir, input_label_dir, filename, PROCESSED_IMAGE_DIR, PROCESSED_LABEL_DIR, PLOT_IMAGE_DIR):
    # Construct file paths
    image_path = os.path.join(input_image_dir, filename)
    bbox_txt_path = os.path.join(input_label_dir, os.path.splitext(filename)[0] + '.txt')
    
    # Check if image and label files exist
    if not os.path.isfile(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return

    if not os.path.isfile(bbox_txt_path):
        print(f"Error: Label file {bbox_txt_path} does not exist.")
        return

    # Load image and label data
    try:
        image = load_image(image_path)
        bbox_data = read_bbox_coordinates(bbox_txt_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    if bbox_data is None:
        print(f"No bounding box data found in {bbox_txt_path}.")
        return
    
    class_id, original_width, original_height, x_tl, y_tl, x_br, y_br = extract_bbox_data(bbox_data)
    cropped_image, crop_x = crop_image(image, original_width, original_height, x_tl, x_br)
    
    save_image(cropped_image, os.path.join(PROCESSED_IMAGE_DIR, filename))
    
    voc_x_tl = max(0, x_tl - crop_x)
    voc_y_tl = y_tl
    voc_x_br = min(cropped_image.shape[1], x_br - crop_x)
    voc_y_br = y_br
    
    img_to_plot = draw_bounding_box(cropped_image.copy(), voc_x_tl, voc_y_tl, voc_x_br, voc_y_br)
    
    save_image(img_to_plot, os.path.join(PLOT_IMAGE_DIR, filename))
    
    my_voc_box = [voc_x_tl, voc_y_tl, voc_x_br, voc_y_br]
    yolo_bbox = convert_bbox_to_yolo(my_voc_box, cropped_image.shape[1], cropped_image.shape[0])
    
    save_yolo_labels(class_id, yolo_bbox.raw_values, os.path.join(PROCESSED_LABEL_DIR, os.path.splitext(filename)[0] + '.txt'))
    
    confirm_yolo_conversion(yolo_bbox, my_voc_box)


# Main function to process all images in the folders
def process_folders(image_dirs, label_dirs, PROCESSED_IMAGE_DIR, PROCESSED_LABEL_DIR, PLOT_IMAGE_DIR):
    for input_image_dir, input_label_dir in zip(image_dirs, label_dirs):
        for filename in os.listdir(input_image_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                process_image(input_image_dir, input_label_dir, filename, PROCESSED_IMAGE_DIR, PROCESSED_LABEL_DIR, PLOT_IMAGE_DIR)

# Run the main function
process_folders([ORIGINAL_IMAGE_DIR], [ORIGINAL_LABEL_DIR], PROCESSED_IMAGE_DIR, PROCESSED_LABEL_DIR, PLOT_IMAGE_DIR)































# import cv2
# import os
# from pybboxes import BoundingBox
# import random
# import numpy as np

# # Directories
# output_image_dir = 'training_data/images'
# output_label_dir = 'training_data/labels'
# plots_dir = 'screenshots/plots'
# input_dir = 'screenshots/original'

# ##### REGULAR CROP + LABELS #####
# # Iterate over the files in the folder
# for filename in os.listdir(input_dir):
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Load image
#         image_path = os.path.join(input_dir, filename)
#         image = cv2.imread(image_path)

#         # Read bounding box coordinates from the text file
#         filename_no_ext = os.path.splitext(filename)[0]
#         bbox_txt_path = os.path.join(input_dir, f"{filename_no_ext}.txt")
#         try:
#             with open(bbox_txt_path, 'r') as file:
#                 bbox_data = file.readline().strip().split()
#         except FileNotFoundError:
#             print(f"Text file not found for image: {filename}")
#             continue

#         # Extract VOC bounding box coordinates
#         # Example content: 0 960 540 432 93 515 209
#         class_id = int(bbox_data[0])
#         original_width = int(bbox_data[1])
#         original_height = int(bbox_data[2])
#         x_tl = int(bbox_data[3])
#         y_tl = int(bbox_data[4])
#         x_br = int(bbox_data[5])
#         y_br = int(bbox_data[6])

#         # New square dimensions (width = height = 540)
#         new_width = original_height

#         # Calculate the horizontal crop offset to fit the bounding box within the new square
#         image_height, image_width, _ = image.shape
#         print(f"Original image dimensions: {image_width}x{image_height}")

#         crop_x = (image_width - new_width) // 2

#         # Ensure the bounding box is within the cropped image
#         if x_tl < crop_x:
#             crop_x = x_tl
#         elif x_br > crop_x + new_width:
#             crop_x = x_br - new_width

#         # Crop the image
#         cropped_image = image[:, crop_x:crop_x + new_width, :]
#         new_height, new_width, _ = cropped_image.shape
#         print(f"Cropped image dimensions: {new_width}x{new_height}")


#         ##### Save Cropped Clean Image #####
#         os.makedirs(output_image_dir, exist_ok=True)
#         image_file = os.path.join(output_image_dir, f'{filename_no_ext}.jpg')
#         cv2.imwrite(image_file, cropped_image)
#         print(f'Image saved to {image_file}')


#         # Adjust bounding box for the cropped image
#         voc_x_tl = max(0, x_tl - crop_x)
#         voc_y_tl = y_tl
#         voc_x_br = min(new_width, x_br - crop_x)
#         voc_y_br = y_br

#         img_to_plot = cropped_image.copy()
#         # Draw bounding box on the cropped image
#         cv2.rectangle(img_to_plot, (voc_x_tl, voc_y_tl), (voc_x_br, voc_y_br), (0, 255, 0), 2)
#         print(f"Cropped bounding box in pixel coordinates: ({voc_x_tl}, {voc_y_tl}), ({voc_x_br}, {voc_y_br})")

#         # Display cropped image with bounding box
#         # cv2.imshow(img_to_plot)

#         ##### Save Plotted Image #####
#         # plot_dir = os.path.join(folder_path, 'plots')
#         os.makedirs(plots_dir, exist_ok=True)
#         image_plot_file = os.path.join(plots_dir, f'{filename_no_ext}.jpg')
#         cv2.imwrite(image_plot_file, img_to_plot)
#         print(f'Image saved to {image_plot_file}')


#         # Ensure these are your current bounding box coordinates for the cropped image
#         my_voc_box = [voc_x_tl, voc_y_tl, voc_x_br, voc_y_br]
#         print(f"ORIGINAL VOC BOX: {my_voc_box}")

#         # Convert VOC bounding box to YOLO format using cropped image dimensions
#         voc_bbox = BoundingBox.from_voc(*my_voc_box, image_size=(new_width, new_height))

#         yolo_bbox = voc_bbox.to_yolo()
#         print(f"YOLO BOX: {yolo_bbox}")

#         raw_values = yolo_bbox.raw_values
#         print(f"YOLO RAW: {raw_values}")

#         ##### Save YOLO results to a .txt file #####
#         os.makedirs(output_label_dir, exist_ok=True)
#         label_file = os.path.join(output_label_dir, f'{filename_no_ext}.txt')
#         with open(label_file, 'w') as f:
#             f.write(f'{class_id} {raw_values[0]} {raw_values[1]} {raw_values[2]} {raw_values[3]}\n')
#         print(f'YOLO results saved to {label_file}')

#         ##### Confirm YOLO Conversion #####
#         yolo_image_size = yolo_bbox.image_size
#         print(f"YOLO Image Size: {yolo_image_size}")

#         new_x_tl = yolo_bbox.x_tl
#         new_y_tl = yolo_bbox.y_tl
#         new_x_br = yolo_bbox.x_br
#         new_y_br = yolo_bbox.y_br
#         print(f"Bounding Box: ({new_x_tl}, {new_y_tl}), ({new_x_br}, {new_y_br})")

#         my_yolo_box = raw_values
#         yolo_bbox = BoundingBox.from_yolo(*my_yolo_box, image_size=yolo_image_size)
#         voc_bbox = yolo_bbox.to_voc()
#         print(voc_bbox)
#         voc_values = voc_bbox.raw_values
#         print(f"CONFIRM VOC BOX: {voc_values}")

#         # Compare the original VOC bounding box coordinates with the converted VOC bounding box coordinates
#         if voc_bbox.values == tuple(my_voc_box):
#             print("YOLO conversion is successful!")
#         else:
#             print("YOLO conversion failed!")
#         print()