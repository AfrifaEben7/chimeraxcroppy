import subprocess
import time
import os
import logging

# Configure logging
logging.basicConfig(
    filename="watchdog.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

source_drive = "Z:\\"  # Drive Z
destination_drive = "X:\\"  # Drive X

# Directory names(just place holders)
watch_directory_name = "Ready_for_processing"  # Directory to monitor
ranamed_to_processing = "Processing"  # Temporary name after detection
ranamed_b = "Complete"  # name after copy

watch_directory = os.path.join(source_drive, watch_directory_name)
processing_dir = os.path.join(source_drive, ranamed_to_processing)
complete_dir = os.path.join(source_drive, ranamed_b)

def robo_move(source, destination):
    """
    Runs robocopy to move the contents from the source to the destination.
    """
    result = subprocess.run(['robocopy', source, destination, '/MOV', '/E', '/MT', '/Z'], capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info("Robocopy completed successfully with no issues.")
    elif 1 <= result.returncode < 8:
        logging.warning(f"Robocopy completed with minor issues (return code: {result.returncode}).")
    else:
        logging.error(f"Robocopy failed with return code {result.returncode}:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If it does not exist, it creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}.")
    else:
        logging.info(f"Directory {directory} already exists.")
    return True

def call_lls_crop(source):
    """
    Calls the lls_crop command with the specified source directory.
    """
    command = ['lls_crop', '-s', source]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info("lls_crop completed successfully.")
        return True
    else:
        logging.error(f"lls_crop failed with return code {result.returncode}:\n{result.stdout}\n{result.stderr}")
        return False

def check_for_txt_files(directory):
    """
    Checks if there are any .txt files in the specified directory.
    """
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            return True
    return False

def monitor_and_transfer_files():
    while True:
        # Check if the directory exists on drive Z
        if os.path.exists(watch_directory) and os.listdir(watch_directory):
            logging.info(f"Directory {watch_directory} found and contains files.")
            # Ensure the Processing directory exists
            if ensure_directory_exists(processing_dir):
                # Move files to Processing directory
                if robo_move(watch_directory, processing_dir):
                    # Call lls_crop
                    if call_lls_crop(processing_dir) and check_for_txt_files(processing_dir):
                        
                            if ensure_directory_exists(complete_dir):
                                
                                if robo_move(processing_dir, complete_dir):
                                    logging.info(f"Files moved from {processing_dir} to {complete_dir}.")
        # waiting time
        time.sleep(5)

if __name__ == "__main__":
    logging.info("Starting directory monitor...")
    monitor_and_transfer_files()