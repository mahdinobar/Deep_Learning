import torch
import dlc_practical_prologue as prologue
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ConvModel_1(nn.Module):
    def __init__(self):
        super(ConvModel_1, self).__init__()

        # Convolutional Layer
        self.conv = nn.Sequential(
            # Input -> 2x14x14
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=1),
            # Conv output -> 32x12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Maxpool output -> 32x6x6
            nn.Dropout2d(p=0.4),
        )
        self.linear = nn.Sequential(
            # Input = 32x6x6 = 1152
            nn.Linear(in_features=1152, out_features=64),
            nn.ReLU(),
            # linear output -> 64
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=2),
            nn.Sigmoid(),
            # output -> 2
        )

    def forward(self, x):
        # Convolutional layers
        output = self.conv(x)
        # Flatten
        output = output.view(output.size(0), -1)
        # Linear Layer
        output = self.linear(output)

        return output


class ConvModel_2(nn.Module):
    """
    Class defining model -> inspired by LeNet
    """
    def __init__(self):
        super(ConvModel_2, self).__init__()

        # Convolutional Layers
        self.conv = nn.Sequential(
            # input -> 14x14x2
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=1),
            # conv output -> 12x12x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 6x6x32
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            # conv output -> 4x4x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 2x2x64
            nn.Dropout2d(p=0.4),
        )
        self.linear = nn.Sequential(
            # linear input -> 64x2x2 = 256x1
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            # linear output -> 64
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=2),
            nn.Sigmoid(),
            # output -> 2
        )

    def forward(self, x):
        # Convolutional layers
        output = self.conv(x)
        # Flatten
        output = output.view(output.size(0), -1)
        # Linear Layer
        output = self.linear(output)

        return output


def train(loader, model, criterion, optimizer, nb_batch):
    """
    Train model
    :param loader:
    :param model:
    :param crierion:
    :param optimizer:
    :param nb_batch:
    :return:
    """
    losses = 0.
    correct = 0
    total = 0

    model.train()

    for batch_idx, (inputs, labels) in enumerate(loader, 0):
        inputs = Variable(inputs)
        labels = Variable(labels)

        # Prediction
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)
        losses += loss.detach().item()

        # Zero parameter gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Train accuracy
        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    train_loss = losses / nb_batch
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy


def validation(loader, model, criterion, nb_batch):
    """
    Validate model
    :param loader:
    :param model:
    :param criterion:
    :param nb_batch:
    :return:
    """
    losses = 0.
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader, 0):
            # Prediction
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)
            losses += loss.detach().item()

            # Accuracy
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        test_loss = losses / nb_batch
        test_accuracy = 100 * correct / total

        return test_loss, test_accuracy


def test(loader, model):
    """
    Compute accuracy of the model
    :param loader:
    :param model:
    :return:
    """
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader, 0):
            # Predication
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (prediction == labels).sum().item()

    return 100 * correct / total


def main():
    """
    Main functoin
    - Load data
    - Create modeal
    - train and evaluate
    :return:
    """
    # ----- PARAMETER --------------------
    nb_pair = 1000
    batch_size = 50
    nb_epochs = 100
    learning_rate = 1e-3
    nb_iteration = 3

    saved_train_accuracy = []
    saved_test_accuracy = []
    for i in range(nb_iteration):
        print('\n------- ITERATION - %d -------' % (i + 1))

        # ----- DATASET --------------------
        train_input, train_target, _, test_input, test_target, _ = prologue.generate_pair_sets(nb_pair)

        # Normalize
        train_input = train_input / 255
        test_input = test_input / 255

        # Split between training (80%) and validation (20%)
        train_dataset = TensorDataset(train_input, train_target)
        train_len = int(0.8 * train_dataset.__len__())
        validation_len = train_dataset.__len__() - train_len
        train_data, validation_data = random_split(train_dataset, lengths=[train_len, validation_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=2)

        # Test
        test_dataset = TensorDataset(test_input, test_target)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # ----- MODEL --------------------
        model = ConvModel_1()
        # model = ConvModel_2(

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # ----- TRAINING + VALIDATION --------------------
        nb_batch_train = train_len // batch_size
        nb_batch_validation = validation_len // batch_size
        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []

        for epoch in range(nb_epochs):
            # TRAIN
            train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, nb_batch_train)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            # VALIDATION
            validation_loss, validation_accuracy = validation(validation_loader, model, criterion, nb_batch_validation)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            # Print progress
            if (epoch + 1) % (nb_epochs / 10) == 0:
                print('Epoch [%d/%d] --- TRAIN: Loss: %.4f - Accuracy: %d%% --- '
                      'VALIDATION: Loss: %.4f - Accuracy: %d%%' %
                      (epoch + 1, nb_epochs, train_loss, train_accuracy, validation_loss, validation_accuracy))

        plt.figure()
        plt.plot(train_losses, label='Train loss')
        plt.plot(validation_losses, label='Validation loss')
        plt.legend(frameon=False)

        # ----- TEST --------------------
        train_accuracy = test(train_loader, model)
        saved_train_accuracy.append(train_accuracy)
        test_accuracy = test(test_loader, model)
        saved_test_accuracy.append(test_accuracy)

        print('Accuracy on train set: %d %%' % train_accuracy)
        print('Accuracy on test set: %d %%' % test_accuracy)

    # ----- MEAN + STD OVER ITERATION --------------------
    print('\nMean train accuracy {:.02f} --- Std train accuracy {:.02f} '
          '\nMean test accuracy {:.02f} --- Std test accuracy {:.02f}'
          .format(torch.FloatTensor(saved_train_accuracy).mean(), torch.FloatTensor(saved_train_accuracy).std(),
                  torch.FloatTensor(saved_test_accuracy).mean(), torch.FloatTensor(saved_test_accuracy).std()))


if __name__ == '__main__':
    main()
    plt.show()
