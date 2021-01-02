import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from path import Path
from PIL import Image

def file_to_array(filename):
    """ Convert Image Files to NP Array """
    temp = np.loadtxt(filename)
    pixel = temp[0].size
    size = int(temp.size / pixel / pixel)
    data = temp.reshape(size, pixel, pixel)
    data = torch.from_numpy(data)
    data = data.reshape(data.shape[0], 1, 1, data.shape[1], data.shape[2])
    return data

def learn_graph(train_e, test_e, learning_graph):
    """ Save the Plot of Learning Error By Interation """
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
    """ Save Learning Result and Network Structure to Report """
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

def save_hidden(x, y, x_test_normal, y_test_normal, x_test_illu, train_h, test_h, illu_h, net):
    """ Save All Hidden States """
    image(x, y, train_h, net)
    image(x_test_normal, y_test_normal, test_h, net)
    image(x_test_illu, y_test_normal, illu_h, net, target=False)
    return

def image(x, y, file, net, target=True):
	""" Save Network Hidden States as Images """
    number = x.shape[0]
    if target is True:
        col = 10
    else:
        col = 9
    # make plots
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
        # different format for targets
        if target is True:
            axs[i, 9].imshow(y[i].reshape(32, 32).detach().numpy(), cmap="gray")
            axs[i, 9].axis('off')
    plt.savefig(file)
    return
    