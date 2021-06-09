import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):

    """
    A simple fully connected neural network with variable amount of hidden layers.
    """
    def __init__(self, input_size, hidden_size, output_size):
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


def get_accuracy(loader, model):
    correct = 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x.reshape(x.shape[0], -1)

            logits = model(x)
            _, predictions = logits.max(dim=1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)

        acc = correct / samples * 100
        print(f'Accuracy {correct}/{samples}: {acc:.2f}%')

    model.train()
    return acc


class IrisDataset(Dataset):

    def __init__(self, dataset_path, transform=None):
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
        item = self.X.iloc[idx]
        if self.transform:
            item = self.transform(item)

        item = torch.Tensor(item)
        label = torch.as_tensor(self.y[idx])
        return item, label


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}.")

    dataset = IrisDataset('iris.txt')

    len_train = int(len(dataset)*0.8)
    len_test = len(dataset) - len_train
    train, test = torch.utils.data.random_split(dataset, [len_train, len_test])

    batch_size = 5
    trainloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    input_size = 4  # Number of features
    output_size = 3  # Number of classes
    model = NeuralNetwork(input_size, [5, 5], output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    accuracies = []
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(trainloader):

            data = data.to(device)
            data = data.reshape(data.shape[0], -1)
            targets = targets.to(device)

            preds = model(data)
            loss = criterion(preds, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        accuracies.append(get_accuracy(testloader, model))

    fig, ax = plt.subplots()
    ax.plot(list(range(num_epochs)), accuracies)
    ax.set_title('Accuracy over epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epochs')
    ax.set_ylim([0, None])
    ax.set_xlim([0, num_epochs])
    plt.grid()
    # fig.tight_layout()
    plt.savefig(f'accuracy_{num_epochs}_epochs.png')
    plt.show()
