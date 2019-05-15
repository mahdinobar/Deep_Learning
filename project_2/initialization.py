import math


def xavier_initialization(w, b, activation_type):
    """
    Performs parameter initialization based on the Xaviar method and calculates the
     gain with respect to activation function type
    :param w: weight of the given layer (l)
    :param b: bias vector of the given layer (l)
    :param activation_type: the type of activation function
    :return:
    """
    # calculate GAIN
    if activation_type == 'relu':
        gain = math.sqrt(2.)
    elif activation_type == 'tanh':
        gain = 5.0 / 3.
    else:
        gain = 1.

    xaviar_init = 2. / (w.size(1) + w.size(0))
    s_deviation = gain * math.sqrt(xaviar_init)

    # Normal initialisation
    w.normal_(0, s_deviation)
    b.normal_(0, s_deviation)
