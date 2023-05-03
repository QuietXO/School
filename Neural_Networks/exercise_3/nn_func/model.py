# Import libraries
from . import visual

# Torch libraries
import torch
import torch.nn as nn


# TODO: add learning curve (TensorBoard)
def train_model(model, data_loader, criterion, optimizer, n_epochs, device=torch.device('cpu'),
                print_steps=1, print_epochs=1, loss_acc=4):
    """
    Start the training of the model parameters
    :param model: Model on which the test will be run
    :param data_loader: DataLoader you want to use for testing
    :param criterion: Type of loss you want to use
    :param optimizer: Type of optimization function
    :param n_epochs: Numer of epochs to be done
    :param device: Torch device on which the test will run (processor by default)
    :param print_steps: Number of printed steps per epoch (one per epoch by default)
    :param print_epochs: Print every n-th epoch (every epoch by default)
    :param loss_acc: Accuracy of the printed loss (4 decimal by default)
    :return: Loss
    """

    print('--- Training Started ---', end='')
    # Variables for epoch print
    print_epochs = (n_epochs if print_epochs is None else print_epochs)
    n_total_steps = len(data_loader)
    mean_loss = 0

    model.to(device)
    for epoch in range(n_epochs):
        mean_loss = 0
        loss_counter = 0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            mean_loss += loss.item()
            loss_counter += 1

            # Backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0:
                if (i+1) % int(n_total_steps / print_steps) == 0:
                    print(f'\nEpoch {epoch + 1} / {n_epochs} | Step {i+1} / {n_total_steps} | '
                          f'Loss: {(mean_loss/loss_counter):.{loss_acc}f}', end='')
    print('\n--- Training Finished ---')

    return mean_loss/n_total_steps


def load_model(model, data_loader, classification, device=torch.device('cpu'), save_path=None,
               class_results=True, show_wrongs=False, n_wrongs=5):
    """
    Load an accuracy test for a model and it's classes
    :param model: Model on which the test will be run
    :param data_loader: DataLoader you want to use for testing
    :param classification: List of classes
    :param device: Torch device on which the test will run (processor by default)
    :param save_path: Load model from a save file (None by default)
    :param class_results: Shows individual class accuracy stats (True by default)
    :param show_wrongs: Shows wrong prediction and the labels (False by default)
    :param n_wrongs: Number of wrong examples which will be shown (5 by default)
    :return: Accuracy of the model on given dataset
    """

    # Load model if necessary
    if save_path is not None and device == torch.device('cuda'):
        model.load_state_dict(torch.load(save_path, map_location='cuda'))
    if save_path is not None and device == torch.device('mps'):
        model.load_state_dict(torch.load(save_path, map_location='mps'))
    if save_path is not None and device == torch.device('cpu'):
        model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.to(device)

    # Start Testing
    wrongs = list()
    n_classes = len(classification)
    y_pred, y_true = [], []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(n_classes)]
        n_class_samples = [0 for _ in range(n_classes)]
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                y_true.append(label)
                y_pred.append(pred)
                if label == pred:
                    n_class_correct[label] += 1
                elif show_wrongs and n_wrongs > len(wrongs) and str(images[i][0][0].numpy()) not in wrongs:
                    wrongs.append(str(images[i][0][0].numpy()))
                    print(f'\nClass: {classification[label]} | Predicted: {classification[pred]}')
                    visual.imshow(images[i])
                n_class_samples[label] += 1

        total_accuracy = 100.0 * n_correct / n_samples
        print(f'Accuracy of the Model: {total_accuracy:.2f} %')

        if class_results:
            for i in range(n_classes):
                accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classification[i]}: {accuracy:.2f} %')
    return total_accuracy, y_pred, y_true


def save_model(model, save_path):
    """
    Save the parameters of your model to a file
    :param model: The model you want to save
    :param save_path: Where do you want to save the file
    :return: None
    """
    torch.save(model.state_dict(), save_path)


# Topology Testing
class Model1(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model1, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 12)
        self.fc4 = nn.Linear(12, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Model2(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model2, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Model3(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model3, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model4(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model4, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class Model5(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model5, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Model6(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model6, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Model7(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(Model7, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Activation Function Testing
class ReLUModel3(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(ReLUModel3, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SigmoidModel3(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(SigmoidModel3, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.sigmoid(self.fc1(x))
        x = torch.nn.functional.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class TanhModel3(nn.Module):
    def __init__(self, img_h, img_w, n_classes):
        super(TanhModel3, self).__init__()
        # Variables
        self.img_h = img_h
        self.img_w = img_w

        # Functions
        self.flattened = self.img_h * self.img_w
        self.fc1 = nn.Linear(self.flattened, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, self.flattened)
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
