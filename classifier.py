# classifier.py

import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader
import seaborn as sns
import os
import re

from config import train_batch_size, test_batch_size
from config import second_layer_in, second_layer_out
from config import epochs, sgd_learning_rate, sgd_momentum_learning_rate, adam_learning_rate, momentum

# directory config
output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)

# dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# neural net
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, second_layer_in)             # first layer: dataset image size as in
        self.fc2 = nn.Linear(second_layer_in, second_layer_out)    # second layer: support layer
        self.fc3 = nn.Linear(second_layer_out, 10)      # third layer: digits as out

    def forward(self, x):
        x = x.view(-1, 28 * 28)         # image to array
        x = torch.relu(self.fc1(x))     # first layers relu
        x = torch.relu(self.fc2(x))     # second layer relu
        x = self.fc3(x)                 # third layer output
        return x

model = NeuralNet()

# loss function
criterion = nn.CrossEntropyLoss()

# training and evaluating with optimizer
def train_and_evaluate(optimizer, epochs=5):
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # evaluating
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss {train_losses[-1]:.4f}, Test Loss {test_losses[-1]:.4f}, Accuracy {accuracy:.2f}%")

    return train_losses, test_losses, test_accuracies

# trainings graph
def plot_metrics(train_losses, test_losses, test_accuracies, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.savefig(os.path.join(output_dir, re.sub(r'[^a-zA-Z0-9]', '', re.sub(r'\s+', '_', title)).lower() + "_metrics.png"))
    plt.show()

# confusion matrix graph
def plot_confusion_matrix(title):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Number")
    plt.ylabel("Real Number")
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(output_dir, re.sub(r'[^a-zA-Z0-9]', '', re.sub(r'\s+', '_', title)).lower() + "_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":

    # sgd
    print("Training with SGD...")
    model = NeuralNet()
    optimizer_sgd = optim.SGD(model.parameters(), lr=sgd_learning_rate)
    train_losses, test_losses, test_accuracies = train_and_evaluate(optimizer_sgd, epochs=epochs)
    plot_metrics(train_losses, test_losses, test_accuracies, "SGD")
    plot_confusion_matrix("SGD")

    # sgd and momentum
    print("\nTraining with SGD + Momentum...")
    model = NeuralNet()
    optimizer_sgd_momentum = optim.SGD(model.parameters(), lr=sgd_momentum_learning_rate, momentum=momentum)
    train_losses, test_losses, test_accuracies = train_and_evaluate(optimizer_sgd_momentum, epochs=epochs)
    plot_metrics(train_losses, test_losses, test_accuracies, "SGD + Momentum")
    plot_confusion_matrix("SGD + Momentum")

    # adam
    print("\nTraining with ADAM...")
    model = NeuralNet()
    optimizer_adam = optim.Adam(model.parameters(), lr=adam_learning_rate)
    train_losses, test_losses, test_accuracies = train_and_evaluate(optimizer_adam, epochs=epochs)
    plot_metrics(train_losses, test_losses, test_accuracies, "ADAM")
    plot_confusion_matrix("ADAM")