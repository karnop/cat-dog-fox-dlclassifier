import os
import shutil
from sklearn.model_selection import train_test_split
from loguru import logger

# Function to organize dataset
def split_dataset(data_dir, output_dir):
    try:
        # Check if output directory already exists, if yes, return without doing anything
        if os.path.exists(output_dir):
            logger.info(f"Output directory '{output_dir}' already exists. Skipping dataset split.")
            return
        
        logger.info("Splitting dataset into train, val, and test sets...")
        categories = os.listdir(data_dir)
        for category in categories:
            category_path = os.path.join(data_dir, category)
            images = os.listdir(category_path)

            train, temp = train_test_split(images, test_size=0.2, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            for split, split_images in zip(["train", "val", "test"], [train, val, test]):
                split_dir = os.path.join(output_dir, split, category)
                os.makedirs(split_dir, exist_ok=True)
                for img in split_images:
                    shutil.copy(os.path.join(category_path, img), os.path.join(split_dir, img))

        logger.info("Dataset split successfully.")
    except Exception as e:
        logger.error(f"Error in splitting dataset: {e}")
        raise
