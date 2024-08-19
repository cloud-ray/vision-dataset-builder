# **Custom Image Dataset Creation with Motion Detection and YOLO**

![Create Image Dataset](assets/build-custom-image-dataset-yolo.jpg)

## **Overview**
This repository contains a comprehensive solution for automating the creation of custom image datasets using motion detection and YOLO object detection. The pipeline is designed to optimize resource usage by only processing frames with detected motion, ensuring that you collect only relevant, high-quality images for your computer vision projects.

## **Table of Contents**
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1: Motion Detection and YOLO Object Detection](#step-1-motion-detection-and-yolo-object-detection)
  - [Step 2: Post-Processing Your Captured Data](#step-2-post-processing-your-captured-data)
  - [Step 3: Uploading Your Dataset to Roboflow](#step-3-uploading-your-dataset-to-roboflow)
  - [Step 4: Backing Up Your Dataset to Google Cloud Storage](#step-4-backing-up-your-dataset-to-google-cloud-storage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## **Features**
- **Motion Detection**: Automatically detects motion in a video stream to trigger YOLO object detection.
- **YOLO Object Detection**: Utilizes the YOLO model to detect custom classes in the video stream.
- **Efficient Resource Usage**: The YOLO model is activated only when motion is detected, conserving computational resources.
- **Automated Dataset Creation**: Captures and processes images based on detection criteria, ready for training without manual annotation.
- **Post-Processing**: Crops images, converts bounding boxes to YOLO format, and validates annotations.
- **Roboflow Integration**: Automates the upload of processed data to Roboflow for further refinement and training.
- **Backup to Google Cloud Storage**: Secures your dataset by backing up local files to Google Cloud Storage.

## **Prerequisites**
Before you begin, ensure you have the following:
- A Roboflow Account: [Sign up here](https://roboflow.com).
- A Roboflow Project: [Create a new project](https://docs.roboflow.com/datasets/create-a-project
) in Roboflow to manage your dataset.
- A Roboflow API Key: [Retrieve your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
- An IP Camera Stream: Use a stream from an IP camera, such as the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app from the Google Play Store.

## **Setup**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/cloud-ray/vision-dataset-builder.git
   cd vision-dataset-builder
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Video Source**
   Edit the `config/configs.py` file with your IP camera's stream address:
   ```python
   VIDEO_SOURCE = 'http://192.168.0.24:8080/video'
   ```

## **Usage**
### **Step 1: Motion Detection and YOLO Object Detection**
Start the motion detection and object detection pipeline:
```bash
python app.py
```
- The application will download the YOLO model file on the first run.
- The system displays a foreground mask for visualizing motion detection. YOLO is triggered upon motion detection.

### **Step 2: Post-Processing Your Captured Data**
Process the captured images and annotations:
```bash
python post_processer.py
```
- The script crops images based on bounding boxes, converts them to YOLO format, and validates the conversion.

### **Step 3: Uploading Your Dataset to Roboflow**
Upload your processed dataset to Roboflow:
```bash
python roboflow_uploader.py
```
- The dataset is saved to your Roboflow project, ready for review and further labeling.

### **Step 4: Backing Up Your Dataset to Google Cloud Storage**
Secure your dataset by backing up to Google Cloud Storage:
```bash
python google_upload.py
```
- The script uploads your local files to GCS and deletes them locally after successful upload.
- **Note:** _The file will be updated, but it's a starting point._

## **Customization**
You can adjust the following parameters in `config/configs.py` to tailor the pipeline to your specific needs:
- `CONFIDENCE_LEVEL`: Minimum confidence required for object detection.
- `CONSECUTIVE_FRAMES`: Number of consecutive frames an object must appear in.
- `MAX_SCREENSHOTS`: Maximum number of screenshots to capture per object.

## **Contributing**
Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **Contact**
For any questions or comments, please contact [Ray](https://www.linkedin.com/in/raymond-fuorry).