'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv_layer1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool_layer1 = nn.MaxPool2d((2,2), stride=2)
        self.flatten = nn.Flatten()

        self.fc_layer1 = nn.Linear(in_features=4608, out_features=500)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''Forward propagation'''
        x = self.relu(self.conv_layer1(x))
        x = self.relu(self.conv_layer2(x))
        x = self.pool_layer1(x)
        x = self.fc_layer1(self.flatten(x))
        y_pred = self.softmax(x)

        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                self.data['train_loss'].append(train_loss.item())

                test_y_true = torch.LongTensor(self.data['test']['y'])
                test_y_pred = self.forward(torch.FloatTensor(np.array(self.data['test']['X'])))

                self.data['test_loss'].append(loss_function(test_y_pred, test_y_true).item())

                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

    def save_plot(self):
        epochs = ['100', '200', '300', '400', '500']
        train_mean = self.data['train_loss']
        test_mean = self.data['test_loss']

        plt.plot(epochs, train_mean, color='blue', marker='o', markersize=5, label='Training Loss')
        plt.plot(epochs, test_mean, color='green', marker='+', markersize=5, linestyle='--',
                 label='Validation Loss')
        plt.title('Learning Curve')
        plt.x