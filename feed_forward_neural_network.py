import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = 'cpu'

# to get the same results every time
torch.manual_seed(123)

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# downloading MNIST dataset if not downloaded yet
def download_datset():
    train_data = torchvision.datasets.MNIST(root='data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_data = torchvision.datasets.MNIST(root='data',
                                           train=False,
                                           transform=transforms.ToTensor())

    return train_data, test_data


# Fully connected neural network with one hidden layer
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.h1(x)
        out = self.relu(out)
        out = self.h2(out)
        return out


def training(model, train_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(train_loader):
            # Move tensors to the configured device
            image = image.reshape(-1, 28 * 28).to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def testing(model, test_loader):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label in test_loader:
            image = image.reshape(-1, 28 * 28).to(device)
            label = label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    train_data, test_data = download_datset()

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = FNN(input_size, hidden_size, num_classes).to(device)
    training(model, train_loader)
    testing(model, test_loader)
