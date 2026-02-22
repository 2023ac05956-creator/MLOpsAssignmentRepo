import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from src.model import SimpleCNN
from src.data_preprocessing import prepare_dataloaders

def train():

    train_loader, val_loader, _ = prepare_dataloaders("data/raw")

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_experiment("cats-dogs-classificationclassification")

    with mlflow.start_run():

        mlflow.log_param("epochs", 5)
        mlflow.log_param("optimizer", "Adam")

        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        mlflow.log_metric("val_accuracy", acc)

        torch.save(model.state_dict(), "models/model.pt")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
