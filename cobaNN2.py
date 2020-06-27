import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import os

# inisialisasi data input
dir = "data_train/LBP_R1/"
kategori = ["ka", "ga", "nga", "pa", "ba", "ma", "ta", "da", "na", "ca", "ja", "nya", "ya", "a", "la", "ra", "sa", "wa",
            "ha", "gha"]
features = [];

for y in kategori:
    path = os.path.join(dir, y)
    for img in os.listdir(path):
        im = Image.open(os.path.join(path, img))
        imgs = list(im.getdata())
        imgs = np.asfarray(imgs)
        imgs = (imgs / 255)
        features.append(imgs)

x = np.array(features) / 255
jum_pixel = x.shape[1]  # jumlah pixel
num_w1 = 256
num_w2 = 128
num_w3 = 64
x = x.reshape((-1, jum_pixel)).astype('float32')

# Inisialisasi + normalisasi Label
class_label = 20
variasi = x.shape[0] / class_label
train_label = np.arange(class_label)
train_label = np.repeat(train_label, variasi)

# One Hot Encoding Labels
train_labels_one_hot = np.eye(np.max(train_label) + 1)[train_label].astype('float32')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)  # Turunan fungsi aktivasi

def view_accuracy(acc1, acc2):
    folds = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
             'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10')
    train = acc1
    valid = acc2

    y_pos = np.arange(len(folds))
    y_value = np.arange(0, 110, 10)
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 11))
    bar1 = ax.bar(y_pos - width / 2, train, width, label='Training')
    bar2 = ax.bar(y_pos + width / 2, valid, width, label='Validation')

    ax.set_title('Model Kfold Accuration', fontweight="bold")
    ax.set_xticks(y_pos)
    ax.set_xticklabels(folds, fontweight="bold")
    ax.set_yticks(y_value)
    ax.set_yticklabels(y_value)
    ax.set_ylabel('Mean Accuration (%)', fontweight="bold")
    ax.legend(bbox_to_anchor=(1, 1.1))

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bar1)
    autolabel(bar2)
    plt.savefig(dir + 'hnodes' + str(num_w1) + '/akurasi_model.png', dpi=300)

def saveWB(wM1, wM2, wM3, wM4, bM1, bM2, bM3, bM4, count):
    df = pd.DataFrame({"A": [wM1, bM1], "B": [wM2, bM2], "C": [wM3, bM3], "D": [wM4, bM4]})
    df.to_pickle(dir + 'hnodes' + str(num_w1) + '/model_' + str(count) + '.pkl')

# Model NN
class NeuralNetwork:
    def __init__(self):

        self.lr = 0.01

        self.w1 = np.random.randn(jum_pixel, num_w1)
        self.b1 = np.random.uniform(low=0, high=0.1, size=(1, num_w1))

        self.w2 = np.random.randn(num_w1, num_w2)
        self.b2 = np.random.uniform(low=0, high=0.1, size=(1, num_w2))

        self.w3 = np.random.randn(num_w2, num_w3)
        self.b3 = np.random.uniform(low=0, high=0.1, size=(1, num_w3))

        self.w4 = np.random.randn(num_w3, class_label)
        self.b4 = np.random.uniform(low=0, high=0.1, size=(1, class_label))

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)

        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3)

        z4 = np.dot(self.a3, self.w4) + self.b4
        self.a4 = sigmoid(z4)

    def backprop(self):

        output_errors = self.y - self.a4

        Eo_dsigmoid = output_errors * dsigmoid(self.a4)  # EOdSigmoid

        error_h3 = np.dot(output_errors, self.w4.T)  # error hidden 3
        Eh3_dsigmoid = error_h3 * dsigmoid(self.a3)  # EH3dsigmoid

        error_h2 = np.dot(error_h3, self.w3.T)  # error hidden 2
        Eh2_dsigmoid = error_h2 * dsigmoid(self.a2)  # EH2dsigmoid

        error_h1 = np.dot(error_h2, self.w2.T)  # error hidden 1
        Eh1_dsigmoid = error_h1 * dsigmoid(self.a1)  # EH1dsigmoid

        self.w4 += self.lr * np.dot(self.a3.T, Eo_dsigmoid)
        self.b4 += self.lr * np.sum(Eo_dsigmoid, axis=0, keepdims=True)

        self.w3 += self.lr * np.dot(self.a2.T, Eh3_dsigmoid)
        self.b3 += self.lr * np.sum(Eh3_dsigmoid, axis=0, keepdims=True)

        self.w2 += self.lr * np.dot(self.a1.T, Eh2_dsigmoid)
        self.b2 += self.lr * np.sum(Eh2_dsigmoid, axis=0, keepdims=True)

        self.w1 += self.lr * np.dot(self.x.T, Eh1_dsigmoid)
        self.b1 += self.lr * np.sum(Eh1_dsigmoid, axis=0, keepdims=True)

    def train(self, x, y):

        self.x = np.array(x, ndmin=2)
        self.y = np.array(y, ndmin=2)
        self.feedforward()
        self.backprop()

    def predict(self, data):
        self.x = np.array(data, ndmin=2)
        self.feedforward()
        return self.a4

    def evaluate(self, x, y):
        corrects, wrongs = 0, 0
        for i in range(len(x)):
            res = self.predict(x[i])
            res_max = res.argmax()
            if res_max == y[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


# Train
akurat_train = []
akurat_val = []
total_train = []
total_val = []
mean_valid = []
mean_train = []
count = 0
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, valid_index in kf.split(x):
    model = NeuralNetwork()
    epochs = 1
    x_train, x_valid = x[train_index], x[valid_index]
    y_train, y_valid = train_label[train_index], train_label[valid_index]
    y_train_one_hot, y_test_one_hot = train_labels_one_hot[train_index], train_labels_one_hot[valid_index]
    for epoch in range(epochs):
        print("epoch: ", epoch + 1)
        for i in range(len(x_train)):
            model.train(x_train[i], y_train_one_hot[i])

        corrects, wrongs = model.evaluate(x_train, y_train)
        akurasi_train = np.around((corrects / (corrects + wrongs)), decimals=3) * 100
        print("Training Accruracy: ", akurasi_train, "%")

        corrects, wrongs = model.evaluate(x_valid, y_valid)
        akurasi_val = np.around((corrects / (corrects + wrongs)), decimals=3) * 100
        print("Validation Accruracy: ", akurasi_val, "%")

        akurat_train.append(akurasi_train)  # simpan nilai akurasi train tiap epoch
        akurat_val.append(akurasi_val)  # simpan nilai akurasi validasi tiap epoch

    count = count + 1
    total_train = np.around((np.mean(akurat_train)), decimals=3)  # rata-rata akurasi_train tiap fold
    total_val = np.around((np.mean(akurat_val)), decimals=3)  # rata-rata akurasi_eval tiap fold
    mean_train.append(total_train)
    mean_valid.append(total_val)
    print("============ Model " + str(count) + " ============")
    print("Mean Training Accuracy : ", total_train, "%")
    print("Mean Validation Accuracy : ", total_val, "%")
    print("=================================")
    akurat_train = []  # mengosongkan value untuk fold selanjutnya
    akurat_val = []  # mengosongkan value untuk fold selanjutnya
    saveWB(model.w1, model.w2, model.w3, model.w4, model.b1, model.b2, model.b3, model.b4, count)

print(mean_train)
print(mean_valid)
view_accuracy(mean_train, mean_valid)