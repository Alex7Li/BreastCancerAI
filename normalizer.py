import numpy as np
import csv
from sklearn.preprocessing import normalize


def load_data(random_seed):
    data = np.genfromtxt('data.csv', delimiter=',', encoding="utf8")
    # Remove labels
    data = np.delete(data, 0, axis=0)

    # Randomize Data
    size = len(data)
    np.random.seed(random_seed)
    perm = np.random.permutation(size)
    data = data[perm]
    print(size)

    # Split up data
    train_size = (size * 4) // 5
    features = data[:, 1:train_size]
    labels = np.array(data[:, 0:1])

    # normalize data
    features = normalize(features, axis=0) * np.sqrt(size)

    # read class data
    i = -2
    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        for row in reader:
            i += 1
            if i == -1:
                continue
            if row[0] == "B":
                labels[i] = 1
            else:
                labels[i] = 0
    labels = labels[perm]

    # split data
    train_features = features[1:train_size]
    test_features = features[train_size:size]
    train_labels = labels[1:train_size]
    test_labels = labels[train_size:size]

    train_data = np.concatenate((train_labels, train_features), axis=1)
    test_data = np.concatenate((test_labels, test_features), axis=1)
    np.savetxt("test_data.csv", test_data, delimiter=',')
    np.savetxt("train_data.csv", train_data, delimiter=',')
    return train_features, train_labels, test_features, test_labels


if __name__ == "__main__":
    load_data(1)
