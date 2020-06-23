import numpy as np
from sklearn.model_selection import KFold


x = np.array ([[0,0,255],  [255,0,0], [0,255,0], [0,0,0],
               [255,255,255],[0,255,255], [255,255,0], [1,2,3],
               [255,0,255], [1,1,0],"""[0,1,1], [1,0,0],
               [0,0,1],[1,0,1],[3,2,1], [3,1,2],
               [1,3,2],[2,3,1], [2,1,3], [2,2,2],
               [2,3,4], [4,5,6], [7,8,9], [10,11,12],
               [6,7,8], [8,9,10], [12,13,14], [13,14,15],
               [14,15,16], [16,17,18], [18,19,20], [17,18,19],
               [19,20,21], [20,21,22], [23,24,25], [26,27,28],
               [22,23,24], [24,25,26], [27,28,29], [30,31,32],
               [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 0],
               [255, 255, 255], [0, 255, 255], [255, 255, 0], [1, 2, 3],
               [255, 0, 255], [1, 1, 0], [0, 1, 1], [1, 0, 0],
               [0, 0, 1], [1, 0, 1], [3, 2, 1], [3, 1, 2],
               [1, 3, 2], [2, 3, 1], [2, 1, 3], [2, 2, 2],
               [2, 3, 4], [4, 5, 6], [7, 8, 9], [10, 11, 12],
               [6, 7, 8], [8, 9, 10], [12, 13, 14], [13, 14, 15],
               [14, 15, 16], [16, 17, 18], [18, 19, 20], [17, 18, 19],
               [19, 20, 21], [20, 21, 22], [23, 24, 25], [26, 27, 28],
               [22, 23, 24], [24, 25, 26], [27, 28, 29], [30, 31, 32]"""
               ])

#Inisialisasi + normalisasi Label
class_label = 20
train_label = np.arange(class_label)
train_label = np.repeat(train_label, 4)
y = np.asfarray(train_label)

#One Hot Encoding Labels
train_targets = np.array(y).astype(np.int)
train_labels_one_hot = np.eye(np.max(train_targets) + 1)[train_targets]


#Train
"""model = NeuralNetwork()
epochs = 1

for epoch in range(epochs):
    print("epoch: ", epoch + 1)
    for i in range(len(x_train)):
        model.train(x_train[i], train_labels_one_hot[i])

    corrects, wrongs = model.evaluate(x_train, y_train)
    print("accruracy train: ", corrects / (corrects + wrongs))
"""
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(x):
    epochs = 2
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train_one_hot, y_test_one_hot = train_labels_one_hot[train_index], train_labels_one_hot[test_index]
    print("============")
    for epoch in range(epochs):
        print("epoch: ", epoch + 1)
        for i in range(len(x_train)):
            print(y_train_one_hot[i])

        print("testing : ")
        print(x_test)
    print("============")


"""kf = KFold(n_splits=10, random_state=None, shuffle=True)
epochs = 2
for epoch in range(epochs):
    print("epoch: ", epoch + 1)
    for train_index, test_index in kf.split(a):
        x_train, x_test = a[train_index], a[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_one_hot, y_test_one_hot = train_labels_one_hot[train_index], train_labels_one_hot[test_index]
        for i in range(len(x_train)):
            print(x_train[i])"""
