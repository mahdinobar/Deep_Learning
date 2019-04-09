from data import generate_data, normalization, convert_to_one_hot_labels
from neural_network import Sequential, Linear
from activation_functions import Relu, Tanh
from losses import LossMSE
from optimization import SGD


def training(m, inputs, targets, batch_size, nb_epochs, lr):
    """
    Training function
    :param m: model
    :param inputs: input data
    :param targets:
    :param batch_size:
    :param nb_epochs:
    :param lr: learning rate
    :return:
    """
    criterion = LossMSE()
    optimizer = SGD(m, lr, momentum=0.9)

    for epoch in range(nb_epochs):
        for batch in range(0, inputs.size(0), batch_size):
            output = m.forward(inputs.narrow(0, batch, batch_size))
            loss = criterion.forward(output, targets.narrow(0, batch, batch_size))
            dl = criterion.backward()
            m.backward(dl)
            optimizer.step()
        if epoch % 50 == 0:
            print('Epoch: ', epoch, ' Loss: ', loss)


def compute_error(m, inputs, targets, batch_size):
    """
    Compute number of errors -> model accuracy
    :param m:
    :param inputs:
    :param targets:
    :param batch_size:
    :return:
    """
    nb_errors = 0

    for batch in range(0, inputs.size(0), batch_size):
        output = m.forward(inputs.narrow(0, batch, batch_size))
        _, prediction = output.max(1)

        for i in range(batch_size):
            if targets[batch + i, prediction[i]] < 0:
                nb_errors += 1

    return nb_errors


def main():
    """ Main function """
    # --------------------------------------------------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------------------------------------------------
    nb_samples = 1000

    # Model
    input_dim = 2
    output_dim = 2
    hidden_dim = 25

    # Training
    number_epochs = 250
    learning_rate = 1e-2
    mini_batch_size = 100

    # --------------------------------------------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------------------------------------------
    # Generate data
    train_input, train_label = generate_data(nb_samples)
    train_label = convert_to_one_hot_labels(train_label)

    test_input, test_label = generate_data(nb_samples)
    test_label = convert_to_one_hot_labels(test_label)

    print('Training data dimension: ', train_input.size())
    print('Training labels dimension: ', train_label.size())

    # Normalize data
    train_input = normalization(train_input)
    test_input = normalization(test_input)

    # --------------------------------------------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------------------------------------------
    model = Sequential(
        Linear(input_dim, hidden_dim),
        Tanh(),
        Linear(hidden_dim, hidden_dim),
        Tanh(),
        Linear(hidden_dim, hidden_dim),
        Tanh(),
        Linear(hidden_dim, output_dim)
    )

    # --------------------------------------------------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------------------------------------------------
    training(model, train_input, train_label, batch_size=mini_batch_size, nb_epochs=number_epochs, lr=learning_rate)

    # ERROR
    train_error = compute_error(model, train_input, train_label, mini_batch_size) / train_input.size(0) * 100
    test_error = compute_error(model, test_input, test_label,  mini_batch_size) / test_input.size(0) * 100

    print('Train error {:.02f}% \nTest error {:.02f}%'.format(train_error, test_error))


if __name__ == '__main__':
    main()
