import torch
import dlc_practical_prologue as prologue
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class AuxiliaryModel(nn.Module):
    """
    Auxiliary Model -> inspired by LeNet
    """
    def __init__(self):
        super(AuxiliaryModel, self).__init__()

        self.conv = nn.Sequential(
            # input -> 1x14x14
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1),
            # conv output -> 32x12x12
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 32x6x6
            nn.Dropout2d(p=0.4),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            # conv output -> 64x4x4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 64x2x2
            nn.Dropout2d(p=0.4),
        )

        self.linear_1 = nn.Sequential(
            # linear input -> 64x2x2 = 256x1
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            # linear output -> 64x1
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1),
            # output -> 10x1
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(in_features=20, out_features=2),
            nn.Sigmoid(),
        )

    def forward_once(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.linear_1(output)
        return output

    def forward(self, x):
        out_digits_1 = self.forward_once(x[:, 0, :, :].unsqueeze(1))
        out_digits_2 = self.forward_once(x[:, 1, :, :].unsqueeze(1))

        # Concatenate
        output_digit = torch.cat((out_digits_1, out_digits_2), 1)
        output = self.linear_2(output_digit)

        return out_digits_1, out_digits_2, output


def train(loader, model, criterion, criterion_digit, optimizer, nb_batch):
    """
    Train Model
    :param loader:
    :param model:
    :param criterion:
    :param criterion_digit:
    :param optimizer:
    :param nb_batch:
    :return:
    """
    losses = 0.
    correct = 0
    total = 0

    model.train()

    for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
        inputs = Variable(inputs)
        classes = Variable(classes)
        labels = Variable(labels)

        # Prediction
        outputs_digit_1, outputs_digit_2, outputs = model(inputs)

        # Loss
        loss_1 = criterion_digit(outputs_digit_1, classes[:, 0])
        loss_2 = criterion_digit(outputs_digit_2, classes[:, 1])
        loss_3 = criterion(outputs, labels)
        loss = 0.2 * loss_1 + 0.2 * loss_2 + 0.6 * loss_3
        losses += loss.detach().item()

        # Zero parameter gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accuracy
        _, predication = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predication == labels).sum().item()

    train_loss = losses / nb_batch
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy


def validation(loader, model, criterion, criterion_digit, nb_batch):
    """
    Validate model
    :param loader:
    :param model:
    :param criterion:
    :param criterion_digit:
    :param nb_batch:
    :return:
    """
    losses = 0.
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
            # Prediction
            outputs_digit_1, outputs_digit_2, outputs = model(inputs)

            # Loss
            loss_1 = criterion_digit(outputs_digit_1, classes[:, 0])
            loss_2 = criterion_digit(outputs_digit_2, classes[:, 1])
            loss_3 = criterion(outputs, labels)
            loss = 0.2 * loss_1 + 0.2 * loss_2 + 0.6 * loss_3
            losses += loss.detach().item()

            # Accuracy
            _, predication = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predication == labels).sum().item()

    test_loss = losses / nb_batch
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy


def test(model, loader):
    """
    Compute the accuracy of the model
    :param model: trained model
    :param loader: data loader -> inputs, classes, labels
    :return: accuracy [%]
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
            # prediction
            _, _, outputs = model(inputs)

            _, predication = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predication == labels).sum().item()
    return 100 * correct / total


def main():
    """
        Main function
        - Load data
        - create train and evaluate model
        :return:
    """
    # ----- PARAMETER --------------------
    nb_pair = 1000
    batch_size = 50
    nb_epochs = 300
    learning_rate = 1e-3
    nb_iteration = 3

    saved_train_accuracy = []
    saved_test_accuracy = []
    for i in range(nb_iteration):
        print('\n------- ITERATION - %d -------' % (i + 1))

        # ----- DATASET --------------------
        train_input, train_target, train_class, test_input, test_target, test_class = prologue.generate_pair_sets(
            nb_pair)

        # Normalize
        train_input = train_input / 255
        test_input = test_input / 255

        # Split between training (80%) and validation (20%)
        train_dataset = TensorDataset(train_input, train_class, train_target)
        train_len = int(0.8 * train_dataset.__len__())
        validation_len = train_dataset.__len__() - train_len
        train_data, validation_data = random_split(train_dataset, lengths=[train_len, validation_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=2)

        # Test
        test_dataset = TensorDataset(test_input, test_class, test_target)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # ----- MODEL --------------------
        model = AuxiliaryModel()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss function
        criterion_digit = nn.CrossEntropyLoss()
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
            train_loss, train_accuracy = train(train_loader, model, criterion, criterion_digit, optimizer, nb_batch_train)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            # VALIDATION
            validation_loss, validation_accuracy = validation(validation_loader, model, criterion, criterion_digit, nb_batch_validation)
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
        train_accuracy = test(model, train_loader)
        saved_train_accuracy.append(train_accuracy)
        test_accuracy = test(model, test_loader)
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
