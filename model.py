import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from path import Path
from PIL import Image
from helpers import file_to_array
from helpers import learn_graph
from helpers import write_report
from helpers import save_hidden
from helpers import image

def main():
    # load input parameters
    epoch = int(sys.argv[1])
    lr = float(sys.argv[2])
    moment = float(sys.argv[3])
    train_f = sys.argv[4]
    train_l = sys.argv[5]
    test_norm_f = sys.argv[6]
    test_norm_l = sys.argv[7]
    test_illu_f = sys.argv[8]
    report = sys.argv[9]
    learning_graph = sys.argv[10]
    train_h = sys.argv[11]
    test_h = sys.argv[12]
    illu_h = sys.argv[13]
    # call the kanizsa training program
    kanizsa(epoch, lr, moment, train_f, train_l, test_norm_f, test_norm_l, test_illu_f, report, train_h, test_h, illu_h, learning_graph)
    return


class IllusionNet(nn.Module):
    """ Class of Illusion Network """
    
    def __init__(self, pixel=32):
        """ Initialize networks with default archetecture and weights """
        super(IllusionNet, self).__init__()

        # construct network architecture
        self.conv0 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.recur0 = nn.RNN(input_size=1024, hidden_size=1024, bias=True, nonlinearity='tanh')
        # initialize weight matrics for the recurrent layer
        self.recur0.weight_ih_l0.data.copy_(0.01 * torch.eye(pixel * pixel))
        self.recur0.weight_hh_l0.data.copy_(0.01 * torch.eye(pixel * pixel))
        # define edge detection kernal
        edge = torch.Tensor([[[[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]],
                             [[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]]])
        # define surface gradient kernel
        fill = torch.Tensor([[[[-0.5, 1, 0.5], [-0.5, 1, 0.5], [-0.5, 1, 0.5]],
                              [[-0.5, -0.5, -0.5], [1, 1, 1], [0.5, 0.5, 0.5]]],
                             [[[0.5, 1, -0.5], [0.5, 1, -0.5], [0.5, 1, -0.5]],
                              [[0.5, 0.5, 0.5], [1, 1, 1], [-0.5, -0.5, -0.5]]]])
        # initialize convolution kernels for edge and surface gradient
        with torch.no_grad():
            self.conv0.weight = torch.nn.Parameter(edge)
            self.conv1.weight = torch.nn.Parameter(fill)
    
    def forward(self, x):
        """ Feedforward action of the network """
        c0 = torch.tanh(self.conv0(x))
        c1 = torch.tanh(self.conv1(c0))
        c2 = torch.tanh(self.conv2(c1))
        c3 = torch.tanh(self.conv3(c2))
        c3 = c3.reshape(-1)
        c3 = torch.cat((c3, c3, c3)).reshape(3, 1, -1)
        r0 = self.recur0(c3)
        seq0 = r0[0][0].reshape(1, -1) + 1
        seq1 = r0[0][1].reshape(1, -1) + 1
        seq2 = r0[0][2].reshape(1, -1) + 1
        o = r0[1].reshape(1, -1) + 1
        return [c0, c1, c2, c3[0], seq0, seq1, seq2], o


def learn(epoch, lr, moment, x, y, x_test_normal, y_test_normal, report, learning_graph):
    """ Training
    
    Input Parameters:
    (1) epoch: the number of training iteration
    (2) lr: learning rate
    (3) moment: momentum
    (4) x: training input data
    (5) y: training label
    (7) x_test_normal: test non-illusion image input
    (8) y_test_normal: test non-illusion label
    (9) report: file name of report to save locally
    (10) learning_graph: file name of learning graph to save locally

    Output:
    net: an object of trained neural network
     
     """
    net = IllusionNet().double()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=moment)
    print('Total Epoch {}'.format(epoch))
    # list to save train and test error iteration
    test_e, train_e = [], []
    # loop over epoches
    for e in range(epoch):
        train_loss, test_loss = 0.0, 0.0
        # compute test loss
        for i in range(x_test_normal.shape[0]):
            test_out = net(x_test_normal[i])[1]
            test_l = F.mse_loss(test_out, y_test_normal[i].reshape(1, -1))
            test_loss += test_l.item()
        for i in range(x.shape[0]):
            train_out = net(x[i])[1]
            train_l = F.mse_loss(train_out, y[i].reshape(1, -1))
            train_loss += train_l.item()
        # compute training loss
        train_e.append(train_loss / x.shape[0])
        test_e.append(test_loss / x_test_normal.shape[0])
        # optimization process
        for i in range(x.shape[0]):
            optimizer.zero_grad()
            result_train = net(x[i])
            train_out = result_train[1]
            train_l = F.mse_loss(train_out, y[i].reshape(1, -1))
            train_l.backward()
            optimizer.step()
        # print training results (errors) fpr reference
        print("Epoch {} === Training Error {}".format(e, train_loss / x.shape[0]))
        print(test_loss / x_test_normal.shape[0])
    # save learning graph and report
    learn_graph(train_e, test_e, learning_graph)
    write_report(epoch, lr, moment, train_e, test_e, report, net)
    return net

def kanizsa(epoch, lr, moment, train_f, train_l, test_norm_f, test_norm_l, test_illu_f, report, train_h, test_h, illu_h, learning_graph):
    """ Extensive Testing Result/Graphs
    
    Input Parameters:
    (1) epoch: the number of training iteration
    (2) lr: learning rate
    (3) moment: momentum
    (4) train_f (train_l): training feature (label)
    (5) test_norm_f (test_normal_l): non-illusion test feature (label)
    (7) test_illu_f: illusion test feature
    (8) report: file name of report to save
    (9) train_h, test_h, illu_h: network hidden states
    (10) learning_graph: file name of learning graph to save locally

    """

    # format test data
    x = file_to_array(train_f)
    x_test_normal = file_to_array(test_norm_f)
    x_test_illu = file_to_array(test_illu_f) 
    y = file_to_array(train_l)
    y = y.reshape(y.shape[0], -1)
    y_test_normal = file_to_array(test_norm_l)
    y_test_normal = y_test_normal.reshape(y_test_normal.shape[0], -1)
    # learning and output results
    print("Start Training...")
    illusion = learn(epoch, lr, moment, x, y, x_test_normal, y_test_normal, report, learning_graph)
    print('Finished Training!')
    print('Start Saving Outputs...')
    save_hidden(x, y, x_test_normal, y_test_normal, x_test_illu, train_h, test_h, illu_h, illusion)
    print('Finished!')
    return

if __name__ == "__main__":
    main()
