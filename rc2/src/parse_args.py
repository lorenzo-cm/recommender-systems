import argparse

def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the arguments
    parser.add_argument('--new_model', action='store_true', 
                        help='Indicates if a new model should be used')

    parser.add_argument('--train_all', action='store_true', 
                        help='Indicates to use all data to train')

    parser.add_argument('--targets', action='store_true', 
                        help='Indicates to generate the predictions for the targets')

    # Parse the arguments
    args = parser.parse_args()

    for arg in vars(args):
        if getattr(args, arg):
            print(f'Argument {arg} is active')
    
    return args
