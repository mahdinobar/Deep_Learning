import torch
import dlc_practical_prologue as prologue
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


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
        for batch_idx, (inputs, labels) in enumerate(loader, 0):
            input_1 = inputs[:, 0, :, :].unsqueeze(1)
            input_2 = inputs[:, 1, :, :].unsqueeze(1)
            # prediction
            outputs = model(input_1, input_2)
            _, predication = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predication == labels).sum().item()
    return 100 * correct / total


class SiameseModel(nn.Module):
    """
    Siamese Model
    """
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.conv = nn.Sequential(
            # input -> 14x14x1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1),
            # conv output -> 12x12x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 6x6x32
            nn.Dropout(p=0.4),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            # conv output -> 4x4x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # maxpooling output -> 2x2x64
            nn.Dropout(p=0.4),
        )

        self.linear = nn.Sequential(
            # linear input -> 2x2x64 = 256x1
            nn.Linear(in_features=256, out_features=64),
            nn.Sigmoid(),
            # linear output -> 64x1
        )

        self.output_layer = nn.Linear(in_features=64, out_features=2)
        # output -> 1x1

    def forward_once(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        res = torch.abs(output1 - output2)
        output = self.output_layer(res)
        return output


def train(loader, model, criterion, optimizer, nb_batch, nb_epochs):
    """
    Train the model
    :param loader: data loader -> inputs and labels
    :param model: model to train
    :param criterion: loss function
    :param optimizer: optimizer
    :param nb_batch: number of batch = nb_samples / batch_size
    :param nb_epochs: number of epochs
    :return:
    """
    for epoch in range(nb_epochs):
        losses = 0.
        for batch_idx, (inputs, labels) in enumerate(loader, 0):
            input_1 = Variable(inputs[:, 0, :, :]).unsqueeze(1)
            input_2 = Variable(inputs[:, 1, :, :]).unsqueeze(1)
            label = Variable(labels)

            # Prediction
            output = model(input_1, input_2)

            # Loss
            loss = criterion(output, label)
            losses += loss.detach().item()

            # Zero parameter gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Print progress
            if (epoch + 1) % (nb_epochs/10) == 0:
                if (batch_idx + 1) % nb_batch == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, nb_epochs,
                                                                      batch_idx + 1, nb_batch,
                                                                      losses / nb_batch))


def main():
    """
    Main function
    - Load data
    - create train and evaluate model
    :return:
    """
    # ----- PARAMETER --------------------
    nb_pair = 1000
    batch_size = 10
    nb_epochs = 100
    learning_rate = 1e-4
    nb_iteration = 1

    saved_train_accuracy = []
    saved_test_accuracy = []
    for i in range(nb_iteration):
        print('\n------- ITERATION - %d -------' % (i+1))

        # ----- DATASET --------------------
        train_input, train_target, train_class, test_input, test_target, test_class = prologue.generate_pair_sets(nb_pair)

        # Normalize
        train_input = train_input / 255
        test_input = test_input / 255

        train_dataset = TensorDataset(train_input, train_target)
        test_dataset = TensorDataset(test_input, test_target)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # ----- MODEL --------------------
        model = SiameseModel()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    print('\n Mean train accuracy {:.02f} --- Std train accuracy {:.02f} '
          '\n Mean test accuracy {:.02f} --- Std test accuracy {:.02f}'
          .format(torch.FloatTensor(saved_train_accuracy).mean(), torch.FloatTensor(saved_train_accuracy).std(),
                  torch.FloatTensor(saved_test_accuracy).mean(), torch.FloatTensor(saved_test_accuracy).std()))


if __name__ == '__main__':
    main()
