import main_new_framework
import main_pytorch

def main():
    """
        Sends the order to run the main files for our deep learning framework
        :param
        :return:
        """

    ###Our deep learning framework
    main_new_framework.main()

    ###PyTorch nn framework to test our model, uncomment to run
    #main_pytorch.main()




if __name__ == '__main__':
    main()