import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class BaseModelTrainer:
    def __init__(self, model, train_dataset, test_dataset, device, criterion, learning_rate=0.01, num_epochs=40, batch_size=32):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.y_true = []
        self.y_pred = []
        self.train_losses = []
        self.test_losses = []

    def plot_weight_histograms(self, model_name="Model"):
        """Plot histograms of the model's layer weights."""
        plt.figure(figsize=(12, 5))
        layer_idx = 1
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                plt.subplot(1, 2, layer_idx)
                plt.hist(layer.weight.detach().cpu().numpy().flatten(), bins=50)
                plt.title(f"{model_name} - Layer {layer_idx} Weights")
                layer_idx += 1
        plt.show()

class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(self, model, train_dataset, test_dataset, device, learning_rate=0.01, num_epochs=40):
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, train_dataset, test_dataset, device, criterion, learning_rate, num_epochs)
        self.train_accuracies = []
        self.test_accuracies = []

    def evaluate(self):
        """Evaluate the classification model on the test dataset, returning both test loss and accuracy."""
        self.model.eval()
        running_test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        self.y_true = []
        self.y_pred = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += targets.size(0)

                self.y_true.extend(targets.cpu().numpy())
                self.y_pred.extend(predicted.cpu().numpy())

        avg_test_loss = running_test_loss / len(self.test_loader)
        test_accuracy = 100 * correct_predictions / total_predictions

        classification_report(self.y_true, self.y_pred, target_names=self.train_dataset.classes)

        return avg_test_loss, test_accuracy, classification_report(self.y_true, self.y_pred, target_names=self.train_dataset.classes)

    def train_with_optimizer(self, optimizer):
        """Train the model using an optimizer, tracking both training and test loss and accuracy for each epoch."""
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            self.model.train()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += targets.size(0)

            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100 * correct_predictions / total_predictions
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            avg_test_loss, test_accuracy, _ = self.evaluate()
            self.test_losses.append(avg_test_loss)
            self.test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        self.plot_training_progress()

    def plot_training_progress(self):
        """Plot training and test loss and accuracy over epochs."""
        epochs = range(1, self.num_epochs + 1)

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.test_losses, label="Test Loss", linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label="Train Accuracy")
        plt.plot(epochs, self.test_accuracies, label="Test Accuracy", linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Test Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        
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
        return conf_matrix

       


class RegressionModelTrainer(BaseModelTrainer):
    def __init__(self, model, train_dataset, test_dataset, device, criterion=nn.MSELoss(), learning_rate=0.01, num_epochs=40):
        super().__init__(model, train_dataset, test_dataset, device, criterion, learning_rate, num_epochs)
        self.train_mae = []
        self.test_mae = []

    def train_with_optimizer(self, optimizer):
        
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            y_true_train = []
            y_pred_train = []
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item() * X_batch.size(0)
                
                y_true_train.extend(y_batch.cpu().numpy())
                y_pred_train.extend(predictions.detach().cpu().numpy())
            
            avg_train_loss = epoch_train_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_train_loss)
            self.train_mae.append(mean_absolute_error(y_true_train, y_pred_train))
            
            val_mae_epoch, _, avg_val_loss = self.evaluate()
            self.test_losses.append(avg_val_loss)
            self.test_mae.append(val_mae_epoch)
            
            print(f"Epoch [{epoch}/{self.num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Train MAE: {self.train_mae[-1]:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, Validation MAE: {self.test_mae[-1]:.4f}")

        return self.train_losses, self.test_losses

    def evaluate(self):
      self.model.eval()
      y_true = []
      y_pred = []
      total_loss = 0.0

      with torch.no_grad():
          for X_batch, y_batch in self.test_loader:
              X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
              predictions = self.model(X_batch)
              
              loss = self.criterion(predictions, y_batch)
              total_loss += loss.item() * X_batch.size(0)

              y_true.extend(y_batch.cpu().numpy())
              y_pred.extend(predictions.detach().cpu().numpy())

      mae = mean_absolute_error(y_true, y_pred)
      rmse = root_mean_squared_error(y_true, y_pred)
      avg_loss = total_loss / len(self.test_loader.dataset)

      return mae, rmse, avg_loss

    def plot_predictions(self):
            """Plot true vs. predicted values."""
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_true, self.y_pred, alpha=0.6)
            plt.plot([min(self.y_true), max(self.y_true)], [min(self.y_true), max(self.y_true)], 'r', linewidth=2)
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title("True vs Predicted Values")
            plt.show()

    def plot_metrics(self):
        """Plot Mean Absolute Error and Loss for training and validation data over epochs."""
        epochs = range(1, self.num_epochs + 1)
        
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_mae, label="Train MAE", color="blue")
        plt.plot(epochs, self.test_mae, label="Validation MAE", linestyle='--', color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Training and Validation MAE Over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_losses, label="Training Loss", color="blue")
        plt.plot(epochs, self.test_losses, label="Validation Loss", linestyle='--', color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()