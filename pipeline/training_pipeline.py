import os
import subprocess
from loguru import logger
from datetime import datetime
from components.data_ingestion import split_dataset
from components.dvc_setup import setup_dvc
from components.data_preprocessing import load_data 
from exceptions.exception import handle_exception

# Set up logging
log_filename = f"logs/training_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger.add(log_filename, rotation="1 MB", level="INFO")

# main function
def main():
    raw_data_dir = "data/raw" 
    split_data_dir = "data/split"

    if not os.path.exists(raw_data_dir):
        logger.error(f"Raw data directory '{raw_data_dir}' not found! download the data from the link in readme using kaggle cli")
        return
    
    try:
        logger.info("Entering the training pipeline")
        #step 1: split the dataset
        split_dataset(raw_data_dir, split_data_dir)

        #step 2: Track dataset with DVC
        # setup_dvc(split_data_dir)

        # step 3: data loading 
        # Load data with batch size of 32
        train_loader, val_loader, test_loader = load_data(split_data_dir, batch_size=32)

    except Exception as e:
        handle_exception(e)

if __name__ == "__main__":
    main()
    


