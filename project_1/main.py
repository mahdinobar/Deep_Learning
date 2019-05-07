import torch
import dlc_practical_prologue as prologue
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(model):
    """
    Weights initialization
    :param m:
    :return:
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.xavier_normal_(model.weight)
        # nn.init.constant_(model.bias, 0.01)

    # elif classname.find('Linear') != -1:
        # nn.init.normal_(model.weight.data, 1.0, 0.02)
        # nn.init.constant_(model.bias.data, 0)


def compute_accuracy(model, loader):
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
            outputs = model(inputs)
            _, predication = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predication == labels).sum().item()
    return 100 * correct / total


def train(loader, model, criterion, optimizer, nb_batch, nb_epochs=1):
    """
    Train the model
    :param loader: data loader -> inputs, classes, labels
    :param model: model to train
    :param criterion: criterion
    :param optimizer: optimizer
    :param nb_batch: number of batch = nb_samples / batch_size
    :param nb_epochs: number of epochs
    :return:
    """
    for epoch in range(nb_epochs):
        losses = 0.
        for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
            inputs = Variable(inputs).to(device)
            classes = Variable(classes).to(device)
            labels = Variable(labels).to(device)

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

        if (epoch + 1) % (nb_epochs / 10) == 0:
            print('Epoch [%d/%d] --- Loss: %.4f' % (epoch + 1, nb_epochs, losses / nb_batch))


class ConvModel(nn.Module):
    """
    Class defining the model
    """
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = nn.Sequential(
            # input -> 14x14x2
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=1),
            # conv output -> 12x12x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 6x6x32
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            # conv output -> 4x4x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 2x2x64
            nn.Dropout2d(p=0.5),
        )

        self.linear = nn.Sequential(
            # linear input -> 2x2x64 = 256x1
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            # linear output -> 64x1

            nn.Linear(in_features=64, out_features=2),
            nn.Sigmoid(),
            # output -> 2x1
        )

    def forward(self, x):
        # Convolutions layers
        output = self.conv(x)
        # Flatten
        output = output.view(output.size(0), -1)
        # Linear layers
        output = self.linear(output)

        return output


def main():
    """
    Main function
    - load data
    - train model
    - evaluate model
    :return:
    """
    # ----- PARAMETER --------------------
    nb_pair = 1000
    batch_size = 100
    nb_epochs = 300
    learning_rate = 1e-4
    nb_iterations = 2

    saved_train_accuracy = []
    saved_test_accuracy = []
    for i in range(nb_iterations):
        print('\n------- ITERATION - %d -------' % (i + 1))

        # ----- DATASET --------------------
        train_input, train_target, train_class, test_input, test_target, test_class = prologue.generate_pair_sets(nb_pair)

        # Normalize
        train_input = train_input/255
        test_input = test_input/255

        train_dataset = TensorDataset(train_input, train_class, train_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        test_dataset = TensorDataset(test_input, test_class, test_target)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # ----- MODEL --------------------
        model = ConvModel()
        # Initialize weights
        # model.apply(weights_init)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # ----- TRAINING --------------------
        nb_batch = nb_pair // batch_size
        train(train_loader, model, criterion, optimizer, nb_batch, nb_epochs=nb_epochs)

        # ----- EVALUATION --------------------
        train_accuracy = compute_accuracy(model, train_loader)
        saved_train_accuracy.append(train_accuracy)
        test_accuracy = compute_accuracy(model, test_loader)
        saved_test_accuracy.append(test_accuracy)
        print('Accuracy on train set: %d %%' % train_accuracy)
        print('Accuracy on test set: %d %%' % test_accuracy)

    # ----- FINAL PERFORMANCES --------------------
    print('\n Mean train accuracy {:.02f} --- Std train accuracy {:.02f} '
          '\n Mean test accuracy {:.02f} --- Std test accuracy {:.02f}'
          .format(torch.FloatTensor(saved_train_accuracy).mean(), torch.FloatTensor(saved_train_accuracy).std(),
                  torch.FloatTensor(saved_test_accuracy).mean(), torch.FloatTensor(saved_test_accuracy).std()))


if __name__ == '__main__':
    main()
