import os
import subprocess
from loguru import logger
from datetime import datetime
from components.data_ingestion import split_dataset
from components.dvc_setup import setup_dvc
from components.data_preprocessing import load_data 
from exceptions.exception import handle_exception
from components.model_architecture import CNNModel
from components.model_training import train_model
from components.model_evaluation import evaluate_model_on_test
from artifacts.artifacts import save_model_path, raw_data_dir, split_data_dir
import torch

# Set up logging
log_filename = f"logs/training_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger.add(log_filename, rotation="1 MB", level="INFO")

# main function
def main():

    if not os.path.exists(raw_data_dir):
        logger.error(f"Raw data directory '{raw_data_dir}' not found! download the data from the link in readme using kaggle cli")
        return
    
    try:
        logger.info("Entering the training pipeline")
        #step 1: split the dataset
        # split_dataset(raw_data_dir, split_data_dir)

        #step 2: Track dataset with DVC
        # setup_dvc(split_data_dir)

        # step 3: data loading 
        # Load data with batch size of 32
        train_loader, val_loader, test_loader = load_data(split_data_dir, batch_size=32)

        # step 4: model 
        model = CNNModel(num_classes=3)

        # step 5: training model
        trained_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

        # step 6: Saving the model
        torch.save(trained_model.state_dict(), save_model_path)
        logger.info(f"Model saved at'{save_model_path}'")

        # step 7: model evaluation
        model.load_state_dict(torch.load(save_model_path))
        accuracy, report = evaluate_model_on_test(model, test_loader)



    except Exception as e:
        handle_exception(e)

if __name__ == "__main__":
    main()
    


