import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from path import Path
from PIL import Image


def main():

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

    kanizsa(epoch, lr, moment, train_f, train_l, test_norm_f, test_norm_l, test_illu_f, report, train_h, test_h, illu_h, learning_graph)

    return


class IllusionNet(nn.Module):

    def __init__(self, pixel=32):
        super(IllusionNet, self).__init__()

        self.conv0 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.recur0 = nn.RNN(input_size=1024, hidden_size=1024, bias=True, nonlinearity='tanh')

        self.recur0.weight_ih_l0.data.copy_(0.01 * torch.eye(pixel * pixel))
        self.recur0.weight_hh_l0.data.copy_(0.01 * torch.eye(pixel * pixel))

        edge = torch.Tensor([[[[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]],
                             [[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]]])
        fill = torch.Tensor([[[[-0.5, 1, 0.5], [-0.5, 1, 0.5], [-0.5, 1, 0.5]],
                              [[-0.5, -0.5, -0.5], [1, 1, 1], [0.5, 0.5, 0.5]]],
                             [[[0.5, 1, -0.5], [0.5, 1, -0.5], [0.5, 1, -0.5]],
                              [[0.5, 0.5, 0.5], [1, 1, 1], [-0.5, -0.5, -0.5]]]])

        with torch.no_grad():
            self.conv0.weight = torch.nn.Parameter(edge)
            self.conv1.weight = torch.nn.Parameter(fill)

    def forward(self, x):

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


def file_to_array(filename):

    temp = np.loadtxt(filename)
    pixel = temp[0].size
    size = int(temp.size / pixel / pixel)
    data = temp.reshape(size, pixel, pixel)
    data = torch.from_numpy(data)
    data = data.reshape(data.shape[0], 1, 1, data.shape[1], data.shape[2])

    return data


def learn_graph(train_e, test_e, learning_graph):

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.plot(range(len(train_e)), train_e, markerfacecolor='blue', markersize=4, color='skyblue', linewidth=3, label="Train")
    plt.plot(range(len(test_e)), test_e, markerfacecolor='red', markersize=4, color='pink', linewidth=3, label='Test')
    plt.title("Train vs Test AVG MSE", size=20)
    plt.xlabel("Epoch")
    plt.ylabel("AVG MSE")
    plt.legend()
    fig.savefig(learning_graph)

    return


def write_report(epoch, lr, momentum, train_e, test_e, report, net):

    with open(report, "w") as f:
        f.write("Summary Report\n\n" +
                "Model Structure\n" +
                "| Input: 1x32x32\n" +
                "| | Convolution0==Kernel3x3==Line==1x32x32==2x32x32\n" +
                "| | | Convolution1==Kernel3x3==Shape==2x32x32==2x32x32\n" +
                "| | | | Convolution2==Kernel3x3==Surface0==2x32x32==1x32x32\n" +
                "| | | | | Convolution3==Kernel3x3==Surface1==1x32x32==1x32x32\n" +
                "| | | | | | Recurrent==Reconstruction==1024==1024\n" +
                "| | | | | | Out==Depth==MoveUp1\n\n" +
                "Functions\n" +
                "| Loss: MSE\n" +
                "| Activation: tanh\n" +
                "| Optimizer: Stochastic Gradient Descent\n\n" +
                "Parameters\n" +
                "| Epoch-{} Learning Rate-{} Momentum-{}\n\n".format(epoch, lr, momentum) +
                "Final Result\n" +
                "| Train Error: {}\n".format(train_e[-1]) +
                "| Test Error: {}\n\n".format(test_e[-1]) +
                "Weight\n" +
                "Convolution0==Line\n")

        f.writelines(str(np.array(net.conv0.weight.data)) + "\n")
        f.writelines("Convolution1==Shape\n")
        f.writelines(str(np.array(net.conv1.weight.data)) + "\n")
        f.writelines("Convolution2==Surface0\n")
        f.writelines(str(np.array(net.conv2.weight.data)) + "\n")
        f.writelines("Convolution3==Surface1\n")
        f.writelines(str(np.array(net.conv3.weight.data)) + "\n")
        f.writelines("Recurrent==Reconstruction\n")
        f.writelines(str(np.array(net.recur0.weight_ih_l0.data)) + "\n")
        f.writelines(str(np.array(net.recur0.weight_hh_l0.data)))

    return


def learn(epoch, lr, moment, x, y, x_test_normal, y_test_normal, report, learning_graph):

    net = IllusionNet().double()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=moment)

    print('Total Epoch {}'.format(epoch))

    test_e, train_e = [], []

    for e in range(epoch):
        train_loss, test_loss = 0.0, 0.0
        for i in range(x_test_normal.shape[0]):
            test_out = net(x_test_normal[i])[1]
            test_l = F.mse_loss(test_out, y_test_normal[i].reshape(1, -1))
            test_loss += test_l.item()
        for i in range(x.shape[0]):
            train_out = net(x[i])[1]
            train_l = F.mse_loss(train_out, y[i].reshape(1, -1))
            train_loss += train_l.item()

        train_e.append(train_loss / x.shape[0])
        test_e.append(test_loss / x_test_normal.shape[0])

        for i in range(x.shape[0]):
            optimizer.zero_grad()
            result_train = net(x[i])
            train_out = result_train[1]
            train_l = F.mse_loss(train_out, y[i].reshape(1, -1))
            train_l.backward()
            optimizer.step()

        print("Epoch {} === Training Error {}".format(e, train_loss / x.shape[0]))
        print(test_loss / x_test_normal.shape[0])

    learn_graph(train_e, test_e, learning_graph)
    write_report(epoch, lr, moment, train_e, test_e, report, net)

    return net


def save_hidden(x, y, x_test_normal, y_test_normal, x_test_illu, train_h, test_h, illu_h, net):

    image(x, y, train_h, net)
    image(x_test_normal, y_test_normal, test_h, net)
    image(x_test_illu, y_test_normal, illu_h, net, target=False)

    return


def image(x, y, file, net, target=True):
    number = x.shape[0]

    if target is True:
        col = 10
    else:
        col = 9

    fig, axs = plt.subplots(number, col, figsize=(col * 1.5, number))

    for i in range(number):
        result = net(x[i])
        axs[i, 0].imshow(x[i].reshape(32, 32), cmap="gray")
        axs[i, 0].axis('off')
        axs[i, 1].imshow(result[0][0].reshape(64, -1).detach().numpy(), cmap="gray")
        axs[i, 1].axis('off')
        axs[i, 2].imshow(result[0][1].reshape(64, -1).detach().numpy(), cmap="gray")
        axs[i, 2].axis('off')
        axs[i, 3].imshow(result[0][2].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 3].axis('off')
        axs[i, 4].imshow(result[0][3].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 4].axis('off')
        axs[i, 5].imshow(result[0][4].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 5].axis('off')
        axs[i, 6].imshow(result[0][5].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 6].axis('off')
        axs[i, 7].imshow(result[0][6].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 7].axis('off')
        axs[i, 8].imshow(result[1].reshape(32, -1).detach().numpy(), cmap="gray")
        axs[i, 8].axis('off')

        if target is True:
            axs[i, 9].imshow(y[i].reshape(32, 32).detach().numpy(), cmap="gray")
            axs[i, 9].axis('off')

    plt.savefig(file)

    return


def kanizsa(epoch, lr, moment, train_f, train_l, test_norm_f, test_norm_l, test_illu_f, report, train_h, test_h, illu_h, learning_graph):

    x = file_to_array(train_f)
    x_test_normal = file_to_array(test_norm_f)
    x_test_illu = file_to_array(test_illu_f)

    y = file_to_array(train_l)
    y = y.reshape(y.shape[0], -1)
    y_test_normal = file_to_array(test_norm_l)
    y_test_normal = y_test_normal.reshape(y_test_normal.shape[0], -1)

    print("Start Training...")
    illusion = learn(epoch, lr, moment, x, y, x_test_normal, y_test_normal, report, learning_graph)
    print('Finished Training!')
    print('Start Saving Outputs...')
    save_hidden(x, y, x_test_normal, y_test_normal, x_test_illu, train_h, test_h, illu_h, illusion)
    print('Finished!')

    return


if __name__ == "__main__":
    main()
