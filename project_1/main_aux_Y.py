import torch
import dlc_practical_prologue as prologue
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_accuracy(model, loader):
    """
    Compute the accuracy of the model
    :param model: trained model
    :param loader: data loader -> inputs, classes, labels
    :return: accuracy [%]
    """
    correct = 0
    total = 0
    num_elements = len(loader.dataset)
    digit_aux= torch.zeros(num_elements)
    with torch.no_grad():
        for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
            # prediction
            outputs = model(inputs)
            batchSize=loader.batch_size
            start = batch_idx*batchSize
            end = start + batchSize
            #print('out',torch.max(outputs.data, 1))
            _, predication = torch.max(outputs.data, 1)
            #digit_aux_batch=predication
            total += labels.size(0)
            correct += (predication == labels).sum().item()
            digit_aux[start:end] = predication
        
    return 100 * correct / total, digit_aux


def compute_main_accuracy(pred, loader):
    """
    Compute the accuracy of the model
    :param model: trained model
    :param loader: data loader -> inputs, classes, labels
    :return: accuracy [%]
    """
    correct = 0
    total = 0
    num_elements = len(loader.dataset)
    digit_aux= torch.zeros(num_elements)
    with torch.no_grad():
        for batch_idx, (inputs, classes, labels) in enumerate(loader, 0):
            # prediction
            batchSize=loader.batch_size
            start = batch_idx*batchSize
            end = start + batchSize
            #print('out',torch.max(outputs.data, 1))
            predication=pred[start+batch_idx]
            total += labels.size(0)
            correct += (predication == labels).sum().item()
            digit_aux[start:end] = predication
        
    return 100 * correct / total, predication


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
            nn.Dropout(0.2),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class main_Model(nn.Module):
    """
    Class defining the model
    """
    def __init__(self):
        super(main_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.Dropout(0.3),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.Dropout(0.4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #print('x',x)
        x=torch.t(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
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
    nb_epochs = 10
    learning_rate = 5e-3

    # ----- DATASET --------------------
    train_input, train_target_main, train_class, test_input, test_target_main, test_class = prologue.generate_pair_sets(nb_pair)
    #print('isize',train_input.size())
    #print('isize',train_class.size())
    #print('tsize',train_target_main.size())
    #First Digit########################################################################33
    train_target=train_class[:,0]
    test_target=test_class[:,0]


    train_dataset = TensorDataset(train_input, train_class, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(test_input, test_class, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # ----- MODEL --------------------
    model = Model()
    model.train(True)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ----- TRAINING --------------------
    nb_batch = nb_pair // batch_size
    train(train_loader, model, criterion, optimizer, nb_batch, nb_epochs=nb_epochs)

    # ----- EVALUATION --------------------
    model_test=model.train(False)
    train_accuracy,left_digit_from_auxiliary_train = compute_accuracy(model_test, train_loader)
    test_accuracy,left_digit_from_auxiliary_test = compute_accuracy(model_test, test_loader)
    print('Accuracy on train set for left auxiliary digit: %d %%' % train_accuracy)
    print('Accuracy on test set for left auxiliary digit: %d %%' % test_accuracy)


    #Second Digit########################################################################
    train_target=train_class[:,1]
    test_target=test_class[:,1]


    train_dataset = TensorDataset(train_input, train_class, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(test_input, test_class, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # ----- MODEL --------------------
    model = Model()
    model.train(True)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ----- TRAINING --------------------
    nb_batch = nb_pair // batch_size
    train(train_loader, model, criterion, optimizer, nb_batch, nb_epochs=nb_epochs)

    # ----- EVALUATION --------------------
    model_test=model.train(False)
    train_accuracy,right_digit_from_auxiliary_train = compute_accuracy(model_test, train_loader)
    test_accuracy,right_digit_from_auxiliary_test = compute_accuracy(model_test, test_loader)
    print('Accuracy on train set for right auxiliary digit: %d %%' % train_accuracy)
    print('Accuracy on test set for right auxiliary digit: %d %%' % test_accuracy)

    #Main Model########################################################################

    train_input=torch.argmax( torch.stack([left_digit_from_auxiliary_train, right_digit_from_auxiliary_train], 1),dim=1)
    test_input=torch.argmax(torch.stack([left_digit_from_auxiliary_test, right_digit_from_auxiliary_test], 1),dim=1)
    

    #print('isize',train_input.size())
    print('train_input',train_input)
    print('test_input',test_input)
    train_target=train_target_main
    #test_input=test_input
    test_target=test_target_main
    #print('tsize',train_target.size())
    print('train_target',train_target)
    print('test_target',test_target)

    train_dataset = TensorDataset(train_input, train_class, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(test_input, test_class, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # ----- MODEL --------------------
    model = main_Model()
    model.train(True)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ----- TRAINING --------------------
    nb_batch = nb_pair // batch_size
    #train(train_loader, model, criterion, optimizer, nb_batch, nb_epochs=nb_epochs)

    # ----- EVALUATION --------------------
    model_test=model.train(False)

    train_accuracy,_= compute_main_accuracy(train_input, train_loader)
    test_accuracy,_= compute_main_accuracy(test_input, test_loader)
    print('Accuracy on train set final: %d %%' % train_accuracy)
    print('Accuracy on test set final: %d %%' % test_accuracy)



if __name__ == '__main__':
    main()