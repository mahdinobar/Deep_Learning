import main_network
import main_auxiliary
import main_siamese

def main():
    """
        Sends the order to run the main files for the different network architectures
        :param
        :return:
        """

    ###Runs the fully connected network, convolutional network, and deep convolutional network
    ###Runs the fully connected network by default. In order to run the convolutional methods, go to the main_network.py an uncomment the related parts
    main_network.main()

    ###Network with the auxiliary loss, uncomment to run
    #main_auxiliary.main()

    ###Siamese network, uncomment to run (takes more time)
    #main_siamese.main()

if __name__ == '__main__':
    main()