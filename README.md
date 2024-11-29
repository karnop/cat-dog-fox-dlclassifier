# Animal Image Classification Pipeline

This repository contains an end-to-end pipeline for classifying animal images (cats, dogs, and foxes) using PyTorch, FastAPI, and MLOps tools like DVC and MLflow. It demonstrates dataset preparation, model training, evaluation, serving the model through an API, and a user interface using Next.js.

## Features

- **Dataset Splitting**: Splits raw data into `train`, `val`, and `test` sets.
- **Data Preprocessing**: Resizes and normalizes images for model training.
- **Model Training**: Trains a CNN model using PyTorch.
- **Model Evaluation**: Computes accuracy and loss on validation and test sets.
- **Model Serving**: Serves predictions through a FastAPI app.
- **User Interface**:
  - A Next.js-based frontend allows users to upload an image and view predictions.
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

3. Install backend dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:

   ```bash
   cd animal-classification-ui
   npm install
   cd ..
   ```

5. Configure DVC with Google Drive:
   ```bash
   dvc remote add -d gdrive_remote gdrive:1yj9KK9hEbmK-CYGoqLT-9JHdRqY6jSuP
   dvc push
   ```

---

## Directory Structure

```bash
├── app.py                # FastAPI app for serving predictions
├── components/
│   ├── data_ingestion.py     # Handles dataset splitting
│   ├── data_preprocessing.py # Preprocesses data for training
│   ├── dvc_setup.py          # DVC setup
│   ├── model_architecture.py # PyTorch model architecture
│   ├── model_training.py     # Trains and saves the model
│   ├── model_evaluation.py   # Evaluates the trained model
│
├── artifacts/
│   ├── artifacts.py          # Config file for storing paths
├── exceptions/
│   ├── exception.py          # Custom exception handling
├── logs/                     # Run logs
├── mlruns/                   # MLflow run logs
├── models/                   # Trained models
├── notebooks/                # Notebooks for experimentation
├── pipeline/
│   ├── training_pipeline.py  # Orchestrates the pipeline steps
├── animal-classification-ui/ # Next.js UI files
│   ├── app/                  # App Router-based structure
│   ├── pages/                # Default pages
│   ├── public/               # Static assets
├── requirements.txt          # Backend dependencies
├── README.md                 # Project documentation
```

---

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

The API will be available at `http://127.0.0.1:8000`.

### 3. Run the Next.js Frontend

Navigate to the `animal-classification-ui` directory and start the development server:

```bash
cd animal-classification-ui
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## Testing the Application

### Backend API

Use Postman or `curl` to send an image for prediction:

```bash
curl -X POST -F "file=@<image_path>" http://127.0.0.1:8000/predict
```

### Frontend UI

1. Open `http://localhost:3000` in your browser.
2. Upload an image using the provided UI.
3. View the classification result displayed below the upload box.

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
- Integrate cloud storage for production-grade deployments.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Dataset**: [Kaggle - Animal Dataset](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes)

- **Frameworks Used**: PyTorch, FastAPI, Next.js, Tailwind CSS, MLflow, DVC
