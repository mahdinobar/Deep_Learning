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

            if (batch_idx + 1) % nb_batch == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, nb_epochs,
                                                                  batch_idx + 1, nb_batch,
                                                                  losses / nb_batch))


class Model(nn.Module):
    """
    Class defining the model
    """
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(288, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
    nb_epochs = 25
    learning_rate = 2e-3

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
    model = Model()
    print(model)
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
    test_accuracy = compute_accuracy(model, test_loader)
    print('Accuracy on train set: %d %%' % train_accuracy)
    print('Accuracy on test set: %d %%' % test_accuracy)


if __name__ == '__main__':
    main()
