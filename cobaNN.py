import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
        features.append(imgs)

x = np.array(features)/255
jum_pixel = x.shape[1]  # jumlah pixel
num_w1 = 4
num_w2 = 4
x = x.reshape((-1, jum_pixel)).astype('float32')


# inisialisasi data test
dir_test = "data_test/LBP_R1/"
features_test = [];

for h in kategori:
    path = os.path.join(dir_test, h)
    for img in os.listdir(path):
        im = Image.open(os.path.join(path, img))
        imgs = list(im.getdata())
        features_test.append(imgs)

test = np.array(features_test)/255
pixel_test = test.shape[1]  # jumlah pixel
test = test.reshape((-1, pixel_test)).astype('float32')


#Inisialisasi + normalisasi Label
class_label = 20
variasi = x.shape[0]/class_label
train_label = np.arange(class_label)
train_label = np.repeat(train_label, variasi)

#One Hot Encoding Labels
train_labels_one_hot = np.eye(np.max(train_label) + 1)[train_label]

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y) #Turunan fungsi aktivasi

# Menampilkan gambar dan hasil prediksi
def view_classify(img, ps):
    ps = np.squeeze(ps)
    fig, (ax1, ax2) = plt.subplots(figsize=(20,10), ncols=2)
    ax1.imshow(img.reshape(64, 64))
    ax1.set_title(ps.argmax())
    ax1.axis('off')
    ax2.barh(np.arange(20), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(20))
    ax2.set_yticklabels(np.arange(20))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

def view_model(model):
    folds = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
             'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10')
    y_pos = np.arange(len(folds))
    model = model

    plt.bar(y_pos, model, align='center', alpha = 1)
    plt.xticks(y_pos,folds)
    plt.ylabel('Rata-rata Akurasi (%)')
    plt.title('Akurasi validasi model Kfold')
    plt.show()

#Model NN
class NeuralNetwork:
    def __init__(self):

        self.lr = 0.01

        self.w1 = np.random.uniform(low=0.1, high=0.4, size=(jum_pixel, num_w1))
        self.b1 = np.random.uniform(low=0.1, high=0.4, size=(1, num_w1))

        self.w2 = np.random.uniform(low=0.1, high=0.4, size=(num_w1, num_w2))
        self.b2 = np.random.uniform(low=0.1, high=0.4, size=(1, num_w2))

        self.w3 = np.random.uniform(low=0.1, high=0.4, size=(num_w2, class_label))
        self.b3 = np.random.uniform(low=0.1, high=0.4, size=(1, class_label))

    def feedforward(self):

        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)

        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3)

    def backprop(self):

        output_errors = self.y - self.a3

        Eo_dsigmoid = output_errors * dsigmoid(self.a3)  # w3

        error_h2 = np.dot(output_errors, self.w3.T) #error hidden 2
        Eh2_dsigmoid = error_h2 * dsigmoid(self.a2)  # EH2dsigmoid

        error_h1 = np.dot(error_h2, self.w2.T) #error hidden 1
        Eh1_dsigmoid = error_h1 * dsigmoid(self.a1)  # EH1dsigmoid

        self.w3 -= self.lr * np.dot(self.a2.T, Eo_dsigmoid)
        self.b3 -= self.lr * np.sum(Eo_dsigmoid, axis=0, keepdims=True)

        self.w2 -= self.lr * np.dot(self.a1.T, Eh2_dsigmoid)
        self.b2 -= self.lr * np.sum(Eh2_dsigmoid, axis=0, keepdims=True)

        self.w1 -= self.lr * np.dot(self.x.T, Eh1_dsigmoid)
        self.b1 -= self.lr * np.sum(Eh1_dsigmoid, axis=0, keepdims=True)

    def train(self, x, y):

        self.x = np.array(x, ndmin=2)
        self.y = np.array(y, ndmin=2)
        self.feedforward()
        self.backprop()

    def predict(self, data):
        self.x = np.array(data, ndmin=2)
        self.feedforward()
        return self.a3

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

#Train
total = []
mean_validate = []
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(x):
    model = NeuralNetwork()
    epochs = 2
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train_one_hot, y_test_one_hot = train_labels_one_hot[train_index], train_labels_one_hot[test_index]
    for epoch in range(epochs):
        print("epoch: ", epoch + 1)
        for i in range(len(x_train)):
            model.train(x_train[i], y_train_one_hot[i])

        corrects, wrongs = model.evaluate(x_test, y_test)
        akurasi = (corrects / (corrects + wrongs)) * 100
        print("Validation Accruracy: ", akurasi , "%")
        total.append(akurasi)

    total = np.mean(total)
    mean_validate.append(np.mean(total))
    print("=============================")
    print("Mean Validation Accuracy : ", total , "%")
    print("=============================")
    total = []


print(mean_validate)
view_model(mean_validate)

"""for i in range(len(test)):
    prediction = model.predict(test[i])
    view_classify(test[i], prediction.reshape(1, -1))
plt.show()"""


"""for epoch in range(epochs):
    print("epoch: ", epoch + 1)
    for i in range(len(x_train)):
        model.train(x_train[i], train_labels_one_hot[i])

    corrects, wrongs = model.evaluate(x_train, y_train)
    print("accruracy train: ", corrects / (corrects + wrongs))

n = [1,70,16,32,50]
for i in n:
    prediction = model.predict(x_train[i])
    view_classify(x_train[i], prediction.reshape(1, -1))
plt.show()"""

