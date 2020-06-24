import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import os


x = np.array ([[0,0,255],  [255,0,0], [0,255,0], [0,0,0],
               [255,255,255],[0,255,255], [255,255,0], [255,255,0],
               [255,0,255], [255,255,0]])
x = x/255
jum_pixel = x.shape[1]  # jumlah pixel
num_w1 = 4
num_w2 = 4
x = x.reshape((-1, jum_pixel)).astype('float32')

#Inisialisasi + normalisasi Label
class_label = 2
variasi = x.shape[0]/class_label
train_label = np.arange(class_label)
train_label = np.repeat(train_label, variasi)

#One Hot Encoding Labels
train_labels_one_hot = np.eye(np.max(train_label) + 1)[train_label].astype('float32')

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
    fig, ax = plt.subplots(figsize=(6,9))
    folds = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
             'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10')
    y_pos = np.arange(len(folds))
    model = model
    plt.bar(y_pos, model, align='center', color="blue")
    plt.xticks(y_pos,folds)
    for i, v in enumerate(model):
        plt.text(y_pos[i] - 0.25, v + 1.5, str(v), color= "blue", fontweight = "bold")
    plt.ylabel('Rata-rata Akurasi (%)')
    plt.title('Akurasi validasi model Kfold')
    plt.show()
    #plt.savefig('data_train/LBP_R1/akurasi_model.png')

def saveWB(wM1,wM2,wM3,bM1,bM2,bM3,count):
    df = pd.DataFrame({"A": [wM1, bM1], "B": [wM2, bM2], "C": [wM3, bM3]})
    df.to_pickle("data_train/LBP_R1/model_" + str(count) + ".pkl")

#Model NN
class NeuralNetwork:
    def __init__(self):

        self.lr = 0.01

        self.w1 = np.random.uniform(low=0.1, high=0.4, size=(jum_pixel, num_w1)).astype('float32')
        self.b1 = np.random.uniform(low=0.1, high=0.2, size=(1, num_w1)).astype("float32")

        self.w2 = np.random.uniform(low=0.1, high=0.4, size=(num_w1, num_w2)).astype('float32')
        self.b2 = np.random.uniform(low=0.1, high=0.2, size=(1, num_w2)).astype('float32')

        self.w3 = np.random.uniform(low=0.1, high=0.4, size=(num_w2, class_label)).astype('float32')
        self.b3 = np.random.uniform(low=0.1, high=0.2, size=(1, class_label)).astype('float32')

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
akurat = []
total = []
mean_validate = []
count = 0
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(x):
    model = NeuralNetwork()
    epochs = 2
    x_train, x_eval = x[train_index], x[test_index]
    y_train, y_eval = train_label[train_index], train_label[test_index]
    y_train_one_hot, y_test_one_hot = train_labels_one_hot[train_index], train_labels_one_hot[test_index]
    for epoch in range(epochs):
        print("epoch: ", epoch + 1)
        for i in range(len(x_train)):
            model.train(x_train[i], y_train_one_hot[i])

        corrects, wrongs = model.evaluate(x_eval, y_eval)
        akurasi = round((corrects / (corrects + wrongs)), 3)*100
        print("Validation Accruracy: ", akurasi, "%")
        akurat.append(akurasi) #simpan nilai akurasi evaluasi tiap epoch

    total = round((np.mean(akurat)), 3) #rata-rata akurasi tiap fold
    mean_validate.append(total)
    print("=============================")
    print("Mean Validation Accuracy : ", total, "%")
    print("=============================")
    akurat = []
    count = count + 1
    saveWB(model.w1, model.w2, model.w3, model.b1, model.b2, model.b3, count)

view_model(mean_validate)

"""print("///////////")
undf = pd.read_pickle("data_train/LBP_R1/model_9.pkl")
print(undf.iloc[1],'\n')
undf = pd.read_pickle("data_train/LBP_R1/model_10.pkl")
print(undf.iloc[1],'\n')
print("///////////")"""



"""karakter = 'gha'
folder = 'data_train/aksara/'+karakter+'/*.png'
images = []
for filename in glob.glob(folder):
    image = Image.open(filename)
    imgs = image.resize((768, 768))
    images.append(imgs)

for (i,new) in enumerate(images):
    new.save('{}{}{}'.format('data_train/aksara/'+karakter+'/'+karakter,i+1,'.png'))"""