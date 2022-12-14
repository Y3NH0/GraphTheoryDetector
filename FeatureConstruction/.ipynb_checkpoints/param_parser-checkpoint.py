import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Feature Extraction (gpickle2feature) based on graph theory.")

    parser.add_argument('--input-path',
                        nargs='?',
                        default='./test_data/0000dc2f3c8bde2d3b61cd1ba3aa5e839c0a7bf432d2e06a88a7ce3b199453e7',
                        help='input binary file.')

    parser.add_argument('--model',
                        nargs='?',
                        default='rf',
                        help='Select the model(rf, knn, svm, mlp).')

    return parser.parse_args()
