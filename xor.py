# -*- coding: utf-8 -*-
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers, Chain
import chainer.functions  as F
import chainer.links  as L
import sys

batchsize = 2
n_epoch   = 1000
n_units   = 12
N         = 8

data = np.array([
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
], dtype=np.float32)
target = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,0 ], dtype=np.int32)
x_train, x_test = np.split(data,   [N])
y_train, y_test = np.split(target, [N])

# Neural net architecture
# Prepare multi-layer perceptron model
class MLP(Chain):
     def __init__(self):
         super(MLP, self).__init__(
             l1=L.Linear(2, n_units),
             l2=L.Linear(n_units, n_units),
             l3=L.Linear(n_units, 2),
         )

     def __call__(self, x):
         h1 = F.relu(self.l1(x))
         h2 = F.relu(self.l2(h1))
         y  = self.l3(h2)
         return y

class Classifier(Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)

     def __call__(self, x, t):
         y             = self.predictor(x)
         self.loss     = F.softmax_cross_entropy(y, t)
         self.accuracy = F.accuracy(y, t)
         return self.loss

# Setup optimizer
model     = L.Classifier(MLP())
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    for i in xrange(0, N, batchsize):
        x_batch = Variable(x_train[perm[i : i + batchsize]])
        y_batch = Variable(y_train[perm[i : i + batchsize]])

        optimizer.zero_grads()
        loss = model(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss     += loss.data * batchsize
        sum_accuracy += model.accuracy.data * batchsize

    print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)
