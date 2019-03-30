import normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
train_features, train_labels, test_features, test_labels=normalizer.load_data(1,range(30))
decisionTree = DecisionTreeClassifier( max_depth=4)
decisionTree.fit(train_features, train_labels)
labels = decisionTree.predict(test_features)
print(labels)
<<<<<<< Updated upstream:decison_tree.py
tn, fp, fn, tp = confusion_matrix(labels, test_labels).ravel()

# print results
print(str(tn) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(tp))
=======
cm = confusion_matrix(labels, test_labels)
tn, fp, fn, tp = cm.ravel();
>>>>>>> Stashed changes:decisonTree.py
error = ((fp + fn) / (tn + fp + fn + tp)) * 100
print(cm)
print("true negatives: " + str(tn))
print("false positives: " + str(fp))
print("false negatives: " + str(fn))
print("true positives: " + str(tp))
print("Percent Error: " + str(error) + "%")
