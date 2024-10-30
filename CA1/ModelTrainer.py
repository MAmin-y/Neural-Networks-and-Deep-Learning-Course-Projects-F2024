import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaseModelTrainer:
    def __init__(self, model, train_dataset, test_dataset, device, criterion, learning_rate=0.01, lambda_l2=0.0001, num_epochs=40):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.device = device
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.y_true = []
        self.y_pred = []

    def train_manual(self):
        """Train the model manually without using an optimizer."""
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # L2 Regularization
                l2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
                loss += self.lambda_l2 * l2_reg

                self.model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= self.learning_rate * param.grad

                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')

    def train_with_optimizer(self, optimizer):
        """Train the model using an optimizer."""
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')

    def plot_weight_histograms(self, model_name="Model"):
        """Plot histograms of the model's layer weights."""
        plt.figure(figsize=(12, 5))
        layer_idx = 1
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                plt.subplot(1, 2, layer_idx)
                plt.hist(layer.weight.detach().cpu().numpy().flatten(), bins=50)
                plt.title(f"{model_name} - Layer {layer_idx} Weights")
                layer_idx += 1
        plt.show()


class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(self, model, train_dataset, test_dataset, device, learning_rate=0.01, lambda_l2=0.0001, num_epochs=40):
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, train_dataset, test_dataset, device, criterion, learning_rate, lambda_l2, num_epochs)

    def evaluate(self):
        """Evaluate the classification model on the test dataset."""
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                self.y_true.extend(labels.cpu().numpy())
                self.y_pred.extend(predicted.cpu().numpy())
        
        print(classification_report(self.y_true, self.y_pred, target_names=self.train_dataset.classes))
    
    def plot_confusion_matrix(self, model_name="Model"):
        """Plot the confusion matrix for the model's predictions."""

        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.train_dataset.classes, yticklabels=self.train_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"Confusion Matrix for Fashion-MNIST Classification for {model_name}")
        plt.show()
        conf_matrix[range(len(conf_matrix)), range(len(conf_matrix))] = 0
        most_confused = np.unravel_index(conf_matrix.argmax(), conf_matrix.shape)
        print(f'Two classes most confused with each other: {self.train_dataset.classes[most_confused[0]]} and {self.train_dataset.classes[most_confused[1]]}')

       


class RegressionModelTrainer(BaseModelTrainer):
    def __init__(self, model, train_dataset, test_dataset, device, learning_rate=0.01, lambda_l2=0.0001, num_epochs=40):
        criterion = nn.MSELoss()
        super().__init__(model, train_dataset, test_dataset, device, criterion, learning_rate, lambda_l2, num_epochs)

    def evaluate(self):
        """Evaluate the regression model on the test dataset."""
        self.model.eval()
        self.y_true = []
        self.y_pred = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                self.y_true.extend(targets.cpu().numpy())
                self.y_pred.extend(outputs.cpu().numpy())

        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
        print(f'Mean Absolute Error: {mae:.4f}')
        print(f'Root Mean Squared Error: {rmse:.4f}')

    def plot_predictions(self):
            """Plot true vs. predicted values."""
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_true, self.y_pred, alpha=0.6)
            plt.plot([min(self.y_true), max(self.y_true)], [min(self.y_true), max(self.y_true)], 'r', linewidth=2)
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title("True vs Predicted Values")
            plt.show()