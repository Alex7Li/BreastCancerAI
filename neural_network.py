import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import normalizer


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linears = []
        self.relus = []
        for i in range(300):
            self.linears.append(nn.Linear(hidden_size, hidden_size))
            self.relus.append(nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        # sigmoid layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for i in range(300):
            out = self.linears[i](out)
            out = self.relus[i](out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = normalizer.load_data(
        1)

    input_size = 30
    hidden_size = 10
    num_classes = 1
    num_epochs = 300
    batch_size = 38
    learning_rate = .001

    train_features = torch.from_numpy(train_features).double()
    train_labels = torch.from_numpy(train_labels).double()
    test_features = torch.from_numpy(test_features).double()
    test_labels = torch.from_numpy(test_labels).double()

    net = Net(input_size, hidden_size, num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training the FNN Model
    for epoch in range(num_epochs):
        # Load a batch of images with its (index, data, class)
        for i in range(0, len(train_features), batch_size):
            features = Variable(train_features[i:i + batch_size]).float()
            labels = Variable(train_labels[i:i + batch_size]).float()
            # Initialize the hidden weight to all zeros
            optimizer.zero_grad()
            # Forward pass: compute the output class given a image
            outputs = net(features)
            # Compute the loss: difference
            # between the output class and the pre-given label
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes
        print('Epoch [%d/%d]' % (epoch + 1, num_epochs))

    # Testing the FNN Model
    correct = 0
    features = Variable(test_features).float()
    outputs = net(features)
    outputs = outputs.data > .5
    for i in range(len(test_labels)):
        if outputs[i] == int(test_labels[i]):
            correct += 1

    print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / len(test_labels)))

    torch.save(net.state_dict(), 'fnn_model.pkl')
