import subprocess
import time
import os
import re
import logging
DEBUG = False
# Configure logging
logging.basicConfig(
    filename="watchdog.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


source_drive = "Z:\\data\\Macropinocytosis\\2024\\"  # Drive Z

destination_drive = "X:\\DeconSandbox"  # Drive X

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
    command = ['lls_crop', '-s', source, '-m', '-c', '488']
    subprocess.run(command)
    logging.info("lls_crop command executed.")
    return True


def check_for_txt_files(directory):
    """
    Checks if there are any .txt files in the specified directory.
    """
    for file in os.walk(directory):
        if file.endswith(".txt"):
            return True
    return False

def monitor_and_transfer_files():
    while True:
        if DEBUG: print('started while:')
        # Check if the directory exists on drive Z
        if os.path.exists(source_drive) and os.listdir(source_drive):
            logging.info(f"Directory {source_drive} found and contains files.")
            #Get the list of all files and directories
            dir_list = [dir for (dir, subdirs, filenames) in os.walk(source_drive) 
                        if  [True for name in filenames if name.endswith('_Settings.txt')] 
                            and 
                            not [True for name in filenames if name.endswith('_ProcessingLog.txt')]
                        ]
            
            if DEBUG: 
                for dir in  dir_list:
                    print ('Ready to Process: ' + dir +'\n')
            for dir in  dir_list:
                print ('Processing: ' + dir)
                if call_lls_crop(dir):
                    logging.info(f"lls crop successful on {dir}.")

        # waiting time
        time.sleep(5)

if __name__ == "__main__":
    logging.info("Starting directory monitor...")
    monitor_and_transfer_files()