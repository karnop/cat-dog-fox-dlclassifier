from sklearn.metrics import classification_report
import torch
from loguru import logger

# Evaluate the model on the test set
def evaluate_model_on_test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())  # Convert to CPU and add to the list
            predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and add to the list
    
    # Generate a classification report
    report = classification_report(true_labels, predicted_labels, target_names=["cat", "dog", "fox"])
    logger.info(f"Classification Report:\n{report}")
    
    # Calculate accuracy manually for logging purposes
    correct = sum([true == pred for true, pred in zip(true_labels, predicted_labels)])
    total = len(true_labels)
    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, report
