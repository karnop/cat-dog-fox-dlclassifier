# Animal Image Classification Pipeline

This repository contains an end-to-end pipeline for classifying animal images (cats, dogs, and foxes) using PyTorch, FastAPI, and MLOps tools like DVC and MLflow. It demonstrates dataset preparation, model training, evaluation, and serving the model through an API.

---

## Features

- **Dataset Splitting**: Splits raw data into `train`, `val`, and `test` sets.
- **Data Preprocessing**: Resizes and normalizes images for model training.
- **Model Training**: Trains a CNN model using PyTorch.
- **Model Evaluation**: Computes accuracy and loss on validation and test sets.
- **Model Serving**: Serves predictions through a FastAPI app.
- **MLOps Integration**:
  - **DVC**: Tracks dataset and model versions, stores them on Google Drive.
  - **MLflow**: Logs metrics, parameters, and artifacts.

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Create and activate a virtual environment:

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
    pip install -r requirements.txt
   ```

4. Configure DVC with Google Drive:
   ```bash
    dvc remote add -d gdrive_remote gdrive:1yj9KK9hEbmK-CYGoqLT-9JHdRqY6jSuP
    dvc push
   ```

## Directory Structure

```bash
├── app.py                # FastAPI app for serving predictions
├── components/
│   ├── data_ingestion.py     # Handles dataset splitting
│   ├── data_preprocessing.py # Preprocesses data for training
│   ├── dvc_setup.py          # dvc setup
│   ├── model_architecture.py # pytorch model architecture
│   ├── model_training.py     # Trains and saves the model
│   ├── model_evaluation.py   # Evaluates the trained model
│
├── artifacts/
│   ├── artifacts.py          # Config file for storing paths
├── exceptions/
│   ├── exception.py          # Custom exception handling
├── logs/                     # run logs
├── mlruns/                   # mlflow run logs
├── models/                   # trained models
├── notebooks/                # notebooks for experimentations
├── pipeline/
│   ├── training_pipeline.py  # Orchestrates the pipeline steps
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
```

## Usage

### 1. Run the Training Pipeline

Execute the following command to prepare the dataset, preprocess it, and train the model:

```bash
python pipeline/training_pipeline.py
```

### 2. Serve the Model with FastAPI

Run the FastAPI app to make predictions:

```bash
python app.py
```

### 3. Test the API

Use Postman or `curl` to send an image for prediction:

```bash
curl -X POST -F "file=@<image_path>" http://127.0.0.1:8000/predict
```

---

## DVC Usage

### Push Data and Model to Google Drive

```bash
dvc add data/raw
dvc add models/trained_model.pth
dvc push
```

### Pull Data and Model from Google Drive

```bash
dvc pull
```

---

## MLflow Usage

1. Start the MLflow server:
   ```bash
   mlflow ui
   ```
2. View training logs and artifacts at `http://127.0.0.1:5000`.

---

## Future Enhancements

- Add CI/CD pipelines for automated testing and deployment.
- Extend support for more animal categories.
- Integrate with cloud storage for production-grade deployments.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Dataset**: [Kaggle - Animal Dataset](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes)

- **Frameworks Used**: PyTorch, FastAPI, MLflow, DVC
