import os
import csv
from datetime import datetime
from google.cloud import storage

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name, credentials_file):
    try:
        # Initialize the Google Cloud Storage client with the credentials
        storage_client = storage.Client.from_service_account_json(credentials_file)

        # Get the target bucket
        bucket = storage_client.bucket(bucket_name)

        # Upload the file to the bucket
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)

        print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
        return f"gs://{bucket_name}/{destination_blob_name}", True
    except Exception as e:
        print(f"Failed to upload {source_file_path}: {e}")
        return None, False

def upload_files_in_directory(bucket_name, local_dir, gcs_dir, credentials_file, log_writer):
    all_files_uploaded = True

    # Iterate through files in the local directory
    for root, _, files in os.walk(local_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(source_file_path, local_dir)
            destination_blob_name = os.path.join(gcs_dir, relative_path).replace("\\", "/")
            gcs_path, success = upload_to_gcs(bucket_name, source_file_path, destination_blob_name, credentials_file)
            
            # Log details to CSV
            log_writer.writerow({
                'Upload DateTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'GCS Original Image Path': gcs_path if gcs_dir == ORIGINAL_IMAGE_GCS_DIR else '',
                'GCS Original Label Path': gcs_path if gcs_dir == ORIGINAL_LABEL_GCS_DIR else '',
                'GCS Processed Image Path': gcs_path if gcs_dir == PROCESSED_IMAGE_GCS_DIR else '',
                'GCS Processed Label Path': gcs_path if gcs_dir == PROCESSED_LABEL_GCS_DIR else '',
                'Roboflow Status': ''  # Placeholder for Roboflow status if needed
            })

            # If any file fails to upload, mark all_files_uploaded as False
            if not success:
                all_files_uploaded = False

    return all_files_uploaded

def delete_local_files(directory):
    # Iterate through files and delete them
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted local file: {file_path}")

def create_or_update_log_csv(log_csv_path):
    # Create or update the CSV file with header
    file_exists = os.path.isfile(log_csv_path)
    with open(log_csv_path, mode='a', newline='') as file:
        fieldnames = ['Upload DateTime', 'GCS Original Image Path', 'GCS Original Label Path', 
                      'GCS Processed Image Path', 'GCS Processed Label Path', 'Roboflow Status']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        return writer

if __name__ == "__main__":
    # Replace the following variables with your specific values
    BUCKET_NAME = "your-bucket-name"
    CREDENTIALS_FILE = "path/to/your/credentials.json"
    LOG_CSV_PATH = "upload_log.csv"

    # Define directory paths
    ORIGINAL_IMAGE_DIR = 'original_data/images/'
    ORIGINAL_LABEL_DIR = 'original_data/labels/'
    PROCESSED_IMAGE_DIR = 'processed_data/images/'
    PROCESSED_LABEL_DIR = 'processed_data/labels/'

    # Define GCS directory paths
    ORIGINAL_IMAGE_GCS_DIR = 'original_data/images'
    ORIGINAL_LABEL_GCS_DIR = 'original_data/labels'
    PROCESSED_IMAGE_GCS_DIR = 'processed_data/images'
    PROCESSED_LABEL_GCS_DIR = 'processed_data/labels'

    # Create or open the log CSV file
    log_writer = create_or_update_log_csv(LOG_CSV_PATH)

    # Upload files and check if all uploads are successful
    all_files_uploaded = True
    all_files_uploaded &= upload_files_in_directory(BUCKET_NAME, ORIGINAL_IMAGE_DIR, ORIGINAL_IMAGE_GCS_DIR, CREDENTIALS_FILE, log_writer)
    all_files_uploaded &= upload_files_in_directory(BUCKET_NAME, ORIGINAL_LABEL_DIR, ORIGINAL_LABEL_GCS_DIR, CREDENTIALS_FILE, log_writer)
    all_files_uploaded &= upload_files_in_directory(BUCKET_NAME, PROCESSED_IMAGE_DIR, PROCESSED_IMAGE_GCS_DIR, CREDENTIALS_FILE, log_writer)
    all_files_uploaded &= upload_files_in_directory(BUCKET_NAME, PROCESSED_LABEL_DIR, PROCESSED_LABEL_GCS_DIR, CREDENTIALS_FILE, log_writer)

    # Delete local files only if all uploads were successful
    if all_files_uploaded:
        delete_local_files(ORIGINAL_IMAGE_DIR)
        delete_local_files(ORIGINAL_LABEL_DIR)
        delete_local_files(PROCESSED_IMAGE_DIR)
        delete_local_files(PROCESSED_LABEL_DIR)
    else:
        print("Some files failed to upload. Local files were not deleted.")
