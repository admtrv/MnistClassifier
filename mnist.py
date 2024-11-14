import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader

from config import train_batch_size
from config import test_batch_size
from config import second_layer_in
from config import second_layer_out

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
