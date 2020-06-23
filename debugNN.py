import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

np.random.seed(0)
cat = np.array([0,0,255,255,0,0])
dog = np.array([255,255,0,0,255,255])
bird = np.array([255,255,0,0,255,255])

imgs = np.asfarray([cat,dog,bird])
x = imgs/255

jum_pixel = x.shape[1]  # jumlah pixel
num_w1 = 4
num_w2 = 4
x.reshape((-1, jum_pixel)).astype('float32')

#Inisialisasi + normalisasi Label
class_label = 3
train_label = np.arange(class_label)
y = np.asfarray(train_label)


#One Hot Encoding Labels
train_targets = np.array(y).astype(np.int)
train_labels_one_hot = np.eye(np.max(train_targets) + 1)[train_targets]

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y) #Turunan fungsi aktivasi

#Model NN
lr = 0.1
w1 = np.random.rand(jum_pixel, num_w1)
b1 = np.random.rand(1, num_w1)

w2 = np.random.rand(num_w1, num_w2)
b2 = np.random.rand(1, num_w2)

w3 = np.random.rand(num_w2, class_label)
b3 = np.random.rand(1, class_label)
print(w1)
print("===============Init=================")
for epoch in range(5):
    print("epoch: ", epoch + 1)
    print(w1)
    print("===============Init=================")
    z1 = np.dot(x,w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3)

    output_errors = train_labels_one_hot - a3
    Eo_dsigmoid = output_errors * dsigmoid(a3)  # w3

    error_h2 = np.dot(output_errors, w3.T) #error hidden 2
    Eh2_dsigmoid = error_h2 * dsigmoid(a2)  # EH2dsigmoid

    error_h1 = np.dot(error_h2, w2.T) #error hidden 1
    Eh1_dsigmoid = error_h1 * dsigmoid(a1)  # EH1dsigmoid
    aneh = lr * np.dot(x.T, Eh1_dsigmoid)
    print(aneh)
    w1 -= aneh
    b1 -= lr * np.sum(Eh1_dsigmoid, axis=0, keepdims=True)
    w2 -= lr * np.dot(a1.T, Eh2_dsigmoid)
    b2 -= lr * np.sum(Eh2_dsigmoid, axis=0, keepdims=True)

    w3 -= lr * np.dot(a2.T, Eo_dsigmoid)
    b3 -= lr * np.sum(Eo_dsigmoid, axis=0, keepdims=True)
    print(output_errors)
    print("===============Error=================")
    print(w1)
    print("===============Udah update=================")

