import torch
def Initialization(w, b, activation_type):
    """
    Performs parameter initialization based on the Xaviar method and calculates the
     gain with respect to activation function type
    Parameters:
        w: weight of the given layer (l)
        b: bias vector of the given layer (l)
        activation_type: the type of activation function
    Returns:
        -
    """


    N_l = w.size(1)
    N_l_1 = w.size(0)
    Xaviar_init = 2. / (N_l + N_l_1)
    if activation_type == 'relu':
        gain = torch.sqrt(2)
    elif activation_type == 'tanh':
        gain = 5.0 / 3.
    else:
        gain = 1.
    s_deviation = gain * Xaviar_init.sqrt()

    w.normal_(0, s_deviation)
    b.normal_(0, s_deviation)