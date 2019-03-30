import normalizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import confusion_matrix

train_features, train_labels, test_features, test_labels = normalizer.load_data(
    1)
decisionTree = KernelRidge(kernel='laplacian')
decisionTree.fit(train_features, train_labels)
labels = decisionTree.predict(test_features)

labels = labels > .5
tn, fp, fn, tp = confusion_matrix(labels, test_labels).ravel()
error = ((fp + fn) / (tn + fp + fn + tp)) * 100
print("true negatives: " + str(tn))
print("false positives: " + str(fp))
print("false negatives: " + str(fn))
print("false negatives: " + str(tp))
print("Percent Error: " + str(error) + "%")
