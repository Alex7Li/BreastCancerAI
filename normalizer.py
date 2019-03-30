import numpy as np
import csv
from sklearn.preprocessing import normalize


def load_data(random_seed):
    my_data = np.genfromtxt('data.csv', delimiter=',', encoding="utf8")
    # Convert first col to numbers
    i = 0
    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        for row in reader:
            if row[0] == "B":
                my_data[i][0] = 1
            else:
                my_data[i][0] = -1
            i += 1
    my_data = np.delete(my_data, (0), axis=0)
    size = len(my_data)
    my_data = normalize(my_data, axis=0) * np.sqrt(size)

    np.random.seed(random_seed)
    my_data = my_data[np.random.permutation(size)]
    print(size)

    train_size = (size * 4) // 5
    train_data = my_data[0:train_size]
    test_data = my_data[train_size:size]
    train_features = train_data[:, 1:train_size]
    test_features = test_data[:, 1:train_size]
    train_labels = train_data[:, 0:1]
    test_labels = test_data[:, 0:1]
    return train_features, train_labels, test_features, test_labels


if __name__ == "__main__":
    load_data(1)
