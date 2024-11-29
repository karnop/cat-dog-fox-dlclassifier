import subprocess
from loguru import logger

def setup_dvc(data_dir):
    """
    Initializes DVC and tracks the dataset.

    Args:
        data_dir (str): Path to the dataset directory to track with DVC.
    """
    try:
        logger.info("Initializing DVC...")
        subprocess.run(["dvc", "init"], check=True)
        subprocess.run(["dvc", "add", data_dir], check=True)
        # subprocess.run(["git", "add", f"{data_dir}.dvc", ".gitignore"], check=True)
        # subprocess.run(["git", "commit", "-m", "Track dataset with DVC"], check=True)
        logger.info("DVC setup completed.")
    except Exception as e:
        logger.error(f"Error in setting up DVC: {e}")
        raise
