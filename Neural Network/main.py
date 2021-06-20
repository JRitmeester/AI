from typing import Sequence
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: Sequence[int], output_size: int):
        """
        :param input_size: Number of input nodes.
        :param hidden_size: List where each number represents the number of nodes for each layer.
        :param output_size: Number of output nodes.
        """

        super().__init__()
        self.flatten = nn.Flatten()

        layers = [nn.Linear(input_size, hidden_size[0]),
                  nn.ReLU()]

        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size[-1], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


def get_accuracy(loader: DataLoader, model: nn.Module) -> float:
    """

    :param loader: A PyTorch DataLoader with the test features and labels.
    :param model: The trained PyTorch model.
    :return: Accuracy of the model on the test set in the range 0-100%.
    """
    correct = 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x.reshape(x.shape[0], -1)            # Flatten the features into a vector.

            logits = model(x)                    # Make the prediction based on the trained model.
            _, predictions = logits.max(dim=1)   # Get the highest predicted value.
            correct += (predictions == y).sum()  # Add the number of correctly classified elements.
            samples += predictions.size(0)       # Count the total amount of tested elements.

        acc = correct / samples * 100

    model.train()
    return acc


class IrisDataset(Dataset):

    def __init__(self, dataset_path: str, transform=None):
        super(IrisDataset, self).__init__()
        datafile = pd.read_csv(dataset_path, header=None,
                               names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
        self.label_dict = {key: val for val, key in enumerate(set(datafile['class']))}

        self.X = datafile[datafile.columns[:-1]]
        self.y = datafile[datafile.columns[-1]].replace(self.label_dict, regex=True)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X.iloc[idx].values
        if self.transform:
            item = self.transform(item)

        item = torch.Tensor(item)
        label = torch.as_tensor(self.y[idx])
        return item, label

def plot_accuracies(accuracies: Sequence[float], export=True, filename=None):
    fig, ax = plt.subplots()
    ax.plot(list(range(len(accuracies))), accuracies)
    ax.set_title('Accuracy over epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epochs')
    ax.set_ylim([0, 101])
    ax.set_xlim([0, num_epochs])
    if export:
        plt.savefig(filename)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up the neural network model')
    # parser.add_argument("-i", "--input", type=int, metavar='',
    #                     required=True, help='The size of the input layer')
    parser.add_argument("-H", "--hidden", type=int, metavar='', nargs='+',
                        required=True, help='The size(s) of the hidden layer(s)')
    # parser.add_argument("-o", "--output", type=int, metavar='',
    #                     required=True, help='The size of the output layer')
    parser.add_argument("-p", "--path", type=str, metavar='',
                        required=True, help='The relative path of the iris.txt file')
    parser.add_argument("-b", "--batch_size", type=int, metavar='',
                        required=False, help="Specify the batch size during training (default: 5)")
    parser.add_argument("-lr", "--learning_rate", type=float, metavar='',
                        required=False, help="Specifiy the learning rate (default: 0.001)")
    parser.add_argument("-e", "--epochs", type=int, metavar='',
                        required=False, help="Specify the number of epochs (default: 100)")
    parser.add_argument('-c', '--cpu', action="store_true",
                        required=False, help='Override the check for cuda and force usage of the CPU.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-ve', '--verbose_export', action='store_true', required=False,
                       help='Export the training graph with all hyperparameters in the filename')
    group.add_argument('-qe', '--quiet_export', action='store_true',
                       help='Export the training graph with a generic name')
    group.add_argument('-np', '--no_plot', action='store_true', required=False,
                       help="Suppress the loss plot after training")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device {device}.")

    # transform = transforms.Compose([transforms.ToTensor()])  # Only used for images
    dataset = IrisDataset(args.path, transform=None)

    len_train = int(len(dataset) * 0.8)
    len_test = len(dataset) - len_train
    train, test = torch.utils.data.random_split(dataset, [len_train, len_test])

    batch_size = 5 if not args.batch_size else args.batch_size
    trainloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    input_size = 4   # Number of features in X
    hidden_size = args.hidden
    output_size = 3  # Number of classes
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.001 if not args.learning_rate else args.learning_rate
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = 100 if not args.epochs else args.epochs
    accuracies = []
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(trainloader):
            data = data.to(device)
            data = data.reshape(data.shape[0], -1)
            targets = targets.to(device)

            predictions = model(data)
            loss = criterion(predictions, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        accuracy = get_accuracy(testloader, model)

        print(f'[Epoch {epoch+1}/{num_epochs}] Accuracy {accuracy:.2f}%')
        accuracies.append(accuracy)

    if args.quiet_export:
        filename = "results/irisNN.png"
        plot_accuracies(accuracies, export=True, filename=filename)
    elif args.verbose_export:
        filename = f"results/irisNN-epochs-{num_epochs}-learning_rate-{lr}-batch_size-{batch_size}-hidden_layers-{str(hidden_size)}.png"
        plot_accuracies(accuracies, export=True, filename=filename)
    elif not args.no_plot:
        plot_accuracies(accuracies)

