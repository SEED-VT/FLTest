import torch
import torch.nn as nn
import torch.nn.functional as F
from diskcache import Index


LOSS_FUNCTIONS_PyTorch = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
}

OPTIMIZER_PyTorch = {
    'Adam': torch.optim.Adam
}

class LeNet(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        """
        Initialize the LeNet model.

        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
        """
        super(LeNet, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)
        # Average pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Fully connected layer 2
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))               # FC1 -> ReLU
        x = F.relu(self.fc2(x))               # FC2 -> ReLU
        x = self.fc3(x)                       # FC3
        return x


class ConvNet(nn.Module):
    """
    ConvNet from Geiping et al. (Inverting Gradients, NeurIPS 2020).
    """

    def __init__(self, channels=1, num_classes=10, width=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 1 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(1 * width),
            nn.ReLU(),
            nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(36 * width, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def sum_model_weights_pytorch(model):
    return sum(p.sum().item() for p in model.parameters())


def _get_weights_from_cache(model_cache_dir, mname, model, channels):
    cache = Index(model_cache_dir)
    key = f'{mname}-channels{channels}'
    state_dict = cache.get(key)
    if state_dict is None:
        state_dict = model.state_dict()
        cache[key] = state_dict
    return state_dict


def get_pytorch_model(model_name, model_cache_dir, deterministic, channels, seed):
    # seed_every_thing(seed)
    model_name2class = {'LeNet': LeNet, 'ConvNet': ConvNet}
    if deterministic is None or model_cache_dir is None or seed is None:
        raise ValueError(
            "model_cache_dir must be provided when deterministic is True/False. seed value is also required")

    if model_name not in model_name2class:
        raise ValueError("Model is not defined.")

    model = model_name2class[model_name](channels=channels)  # default
    if deterministic:
        state_dict = _get_weights_from_cache(
            model_cache_dir, model_name, model, channels=channels)
        model.load_state_dict(state_dict)

    return model


def train(net, trainloader, epochs, device, loss_fn, opitmzer_name, **args):
    # seed_every_thing(seed=args['seed'])

    # Ensure model is on the correct device
    net.to(device)

    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    optimizer = OPTIMIZER_PyTorch[opitmzer_name](net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        # if verbose:
        #     print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return net, epoch_loss.item()


def test(net, testloader, device, loss_fn, **args):
    # seed_every_thing(args['seed'])

    # Ensure model is on the correct device
    net.to(device)

    criterion = LOSS_FUNCTIONS_PyTorch[loss_fn]()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
