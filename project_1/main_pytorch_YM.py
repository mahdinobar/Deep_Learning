import torch
import dlc_practical_prologue as prologue

import time
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_data, convert_to_one_hot_labels, normalization


def training(m, inputs, targets, batch_size, nb_epochs, lr):
    """
    Training function
    :param m: model
    :param inputs:
    :param targets:
    :param batch_size:
    :param nb_epochs:
    :param lr: learning rate
    :return:
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(m.parameters(), lr=lr, momentum=0.9)

    for epoch in range(nb_epochs):
        for batch in range(0, inputs.size(0), batch_size):
            output = m(inputs.narrow(0, batch, batch_size))
            loss = criterion(output, targets.narrow(0, batch, batch_size))
            m.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch % 50 == 0) or (epoch == nb_epochs-1):
            print('Epoch: {}    Loss: {:.04f}'.format(epoch, loss.item()))


def prediction(m, inputs):
    """
    Prediction output from inputs
    :param m: model
    :param inputs:
    :return:
    """
    output = m(inputs)
    _, predictions = output.max(1)
    return predictions


def compute_error(m, inputs, targets, batch_size):
    """
    Compute number of error -> model accuracy
    :param m:
    :param inputs:
    :param targets:
    :param batch_size:
    :return:
    """
    nb_errors = 0

    for batch in range(0, inputs.size(0), batch_size):
        outputs = prediction(m, (inputs.narrow(0, batch, batch_size)))

        for i in range(batch_size):
            if targets.data[batch + i, outputs[i]] < 1:
                nb_errors += 1

    return nb_errors


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


def main():
    """ Main function """
    # --------------------------------------------------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------------------------------------------------
    nb_samples = 1000
    iteration = 10

    # Model
    input_dim = 2*14*14
    output_dim = 2
    hidden_dim = 30

    # Training
    number_epochs = 200
    learning_rate = 3e-3
    mini_batch_size = 100

    # --------------------------------------------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------------------------------------------
    # Generate data
    # train_input, train_label = generate_data(nb_samples)
    # train_label = convert_to_one_hot_labels(train_label)
    #
    # test_input, test_label = generate_data(nb_samples)
    # test_label = convert_to_one_hot_labels(test_label)
    #
    # print('Training data dimension: ', train_input.size())
    # print('Training labels dimension: ', train_label.size())

    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    train_input=train_input.view(1000, 2 * 14 * 14)
    test_input = test_input.view(1000, 2 * 14 * 14)

    
    # train_target=train_target.float()
    # test_target=test_target.float()


    train_target=(torch.stack((1-train_target, train_target)).t()).float()
    test_target = (torch.stack((1-test_target, test_target)).t()).float()


    # [torch.FloatTensor([train_target).view(1, -1), torch.FloatTensor(train_target_b).view(1, -1)]
        # Normalize data
    # train_input = normalization(train_input)
    # test_input = normalization(test_input)

    saved_train_error = []
    saved_train_time = []
    saved_test_error = []
    saved_prediction_time = []

    for i in range(iteration):
        # --------------------------------------------------------------------------------------------------
        # MODEL
        # --------------------------------------------------------------------------------------------------
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Xavier initialization
        model.apply(init_weights)

        # --------------------------------------------------------------------------------------------------
        # TRAINING
        # --------------------------------------------------------------------------------------------------
        start_train_time = time.time()
        training(model, train_input, train_target, batch_size=mini_batch_size, nb_epochs=number_epochs, lr=learning_rate)
        end_train_time = time.time()

        # ERROR
        train_error = compute_error(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
        saved_train_error.append(train_error)
        test_error = compute_error(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
        saved_test_error.append(test_error)

        # Prediction time
        start_pred_time = time.time()
        for batch in range(0, test_input.size(0), mini_batch_size):
            prediction(model, (test_input.narrow(0, batch, mini_batch_size)))
        end_pred_time = time.time()

        train_time = end_train_time - start_train_time
        saved_train_time.append(train_time)
        prediction_time = end_pred_time - start_pred_time
        saved_prediction_time.append(prediction_time)

        print('\nTrain error {:.02f}% --- Train time {:.02f}s '
              '\nTest error {:.02f}% --- Prediction time {:.08f}s'
              .format(train_error, train_time,
                      test_error, prediction_time))

    print('\nMean train error {:.02f}% --- Mean train time {:.02f}s '
          '\nMean test error {:.02f}% --- Mean prediction time {:.08f}s'
          .format(torch.FloatTensor(saved_train_error).mean(), torch.FloatTensor(saved_train_time).mean(),
                  torch.FloatTensor(saved_test_error).mean(), torch.FloatTensor(saved_prediction_time).mean()))


if __name__ == '__main__':
    main()

print('end')