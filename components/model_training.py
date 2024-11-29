import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from loguru import logger
from tqdm import tqdm

# Training function
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    # Learning rate scheduler (optional, but useful)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero out the gradients from the previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the weights
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  # Step the learning rate scheduler

        # Calculate average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # Evaluate the model on the validation set
        val_accuracy = evaluate_model(model, val_loader, device)
        logger.info(f"Validation Accuracy after epoch {epoch+1}: {val_accuracy:.2f}%")
    
    logger.info("Training complete.")
    return model

# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
