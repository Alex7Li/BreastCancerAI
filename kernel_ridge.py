import normalizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import confusion_matrix

train_features, train_labels, test_features, test_labels = normalizer.load_data(1, range(30))
ridge = KernelRidge(kernel='laplacian')
ridge.fit(train_features, train_labels)
labels = ridge.predict(test_features)

labels = labels > .5
tn, fp, fn, tp = confusion_matrix(labels, test_labels).ravel()

# print results
print(str(tn) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(tp))
error = ((fp + fn) / (tn + fp + fn + tp)) * 100
print("true negatives: " + str(tn))
print("false positives: " + str(fp))
print("false negatives: " + str(fn))
print("true positives: " + str(tp))
print("Percent Error: " + str(error) + "%")
