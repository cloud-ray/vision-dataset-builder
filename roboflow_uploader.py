# roboflow_uploader.py
import os
import roboflow
import uuid
from config.configs import *
from dotenv import load_dotenv

load_dotenv()

def upload_dataset_to_roboflow(api_key, workspace_url, project_name, num_retries=3):
    try:
        # Initialize Roboflow API with the provided API key
        rf = roboflow.Roboflow(api_key=api_key)
        
        # Get the workspace using the workspace URL
        workspace = rf.workspace(workspace_url)
        
        # Generate a unique batch name using a UUID
        roboflow_batch_name = f"sdk_batch_{str(uuid.uuid4())[:8]}"
        
        # Upload the dataset to the Roboflow project
        workspace.upload_dataset(
            dataset_path=PROCESSED_DIR, # (str): path to the dataset
            project_name=project_name, # (str): name of the project
            num_workers=NUM_WORKERS, # (int): number of workers to use for parallel uploads
            dataset_format=DATASET_FORMAT, # (str): format of the dataset (`voc`, `yolov8`, `yolov5`)
            project_license=PROJECT_LICENSE, # (str): license of the project (set to `private` for private projects, only available for paid customers)
            project_type=PROJECT_TYPE, # (str): type of the project (only `object-detection` is supported)
            batch_name=roboflow_batch_name, # (str): name of the batch (optional, default is None)
            num_retries=num_retries # (int): number of retries (optional, default is 0)
        )
        
        print(f"Dataset uploaded with batch name: {roboflow_batch_name}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


upload_dataset_to_roboflow(
    api_key=os.getenv('ROBOFLOW_API_KEY'),
    workspace_url=os.getenv('WORKSPACE_URL'),
    project_name=os.getenv('PROJECT_NAME')
)
