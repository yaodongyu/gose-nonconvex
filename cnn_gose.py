"""
We apply Gose algorithm (https://arxiv.org/abs/1712.03950) to Adam on
CNN to find local minima more efficiently.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse


# Training settings
parser = argparse.ArgumentParser(description='Gose Example')

parser.add_argument('--BATCH_SIZE', type=int, default=100, metavar='N',
                    help='batch size for Adam in training (default: 100)')
parser.add_argument('--BATCH_SIZE_POWER', type=int, default=10, metavar='N',
                    help='batch size for power method in training (default: 10)')
parser.add_argument('--SUBSAMPLE_SIZE', type=int, default=10, metavar='N',
                    help='subsample size (how many mini-batch) for tracking gradient norm (default: 10)')
parser.add_argument('--TRACK_INTERVAL', type=int, default=100, metavar='N',
                    help='interval size for tracking gradient norm (default: 100)')
parser.add_argument('--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate for Adam (default: 0.001)')
parser.add_argument('--LR_POWER', type=float, default=0.5, metavar='LR',
                    help='learning rate for power method (default: 0.5)')
parser.add_argument('--NORM_THRESHOLD', type=float, default=0.1, metavar='LR',
                    help='threshold for gradient norm (default: 0.001)')
parser.add_argument('--EPOCH', type=int, default=100, metavar='LR',
                    help='total epoch (data pass) for the algorithm (default: 100)')
parser.add_argument('--EPOCH_POWER', type=int, default=10, metavar='N',
                    help='epoch for power method (default: 10)')
parser.add_argument('--INNER_ITER_POWER', type=int, default=20, metavar='N',
                    help='inner iteration for power method (default: 20)')
parser.add_argument('--LAMBDA_POWER', type=float, default=5.0, metavar='L',
                    help='normalization term for power method (default: 5.0)')
parser.add_argument('--ETA_NEG', type=float, default=0.5, metavar='L',
                    help='step size for negative curvature descent (default: 0.5)')
parser.add_argument('--NO_CUDA', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.NO_CUDA and torch.cuda.is_available()


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 dataset
# Training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data Loader for Adam in training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE, shuffle=True)
# Data Loader for Power method
train_loader_power = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.BATCH_SIZE_POWER, shuffle=True)
# Data Loader in testing
testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE, shuffle=True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.softplus(self.conv1(x), 5)
        x = self.pool(F.softplus(self.conv2(x), 5))
        x = self.pool(F.softplus(self.conv3(x), 5))
        x = x.view(-1, 64 * 5 * 5)
        x = F.softplus(self.fc1(x), 5)
        x = F.softplus(self.fc2(x), 5)
        x = self.fc3(x)
        return x

    def partial_grad(self, input_data, target, loss_function):
        """
        Function to compute the stochastic gradient
        args : data, target, loss_function
        return loss_partial
        """

        output_data = self.forward(input_data)
        # compute the partial loss

        loss_partial = loss_function(output_data, target)

        # compute gradient
        loss_partial.backward()

        return loss_partial

    def calculate_loss_grad(self, dataset, loss_function, number_batch):
        """
        Function to compute the large-batch loss and the large-batch gradient
        args : dataset, loss function and number of batches
        return : large_batch_loss and large_batch_norm
        """

        large_batch_loss = 0.0
        large_batch_norm = 0.0

        for data_i, data in enumerate(dataset):
            # only calculate the sub-sampled large batch
            if data_i > number_batch - 1:
                break

            inputs, labels = data
            # wrap data and target into variable
            if args.cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            large_batch_loss += (1.0 / number_batch) * self.partial_grad(inputs, labels, loss_function).data[0]

        # calculate the norm of the large-batch gradient
        for param in self.parameters():
            large_batch_norm += param.grad.data.norm(2) ** 2

        large_batch_norm = np.sqrt(large_batch_norm) / number_batch

        print('large batch loss:', large_batch_loss)
        print('large batch norm:', large_batch_norm)

        return large_batch_loss, large_batch_norm

    def power_method(self, dataset_power, loss_function, args):
        """
        Function to calculate the smallest eigenvector v of Hessian matrix H,
            and take a negative curvature descent step along v if v^{T}Hv < 0
        args : dataset, loss function, args
        return : estimate_value, i.e., v^{T}Hv
        """

        # load the parameters from args
        n_epoch = args.EPOCH_POWER
        inner_iteration = args.INNER_ITER_POWER
        learning_rate = args.LR_POWER
        eta_neg = args.ETA_NEG
        lambda_power = args.LAMBDA_POWER

        # record the starting point x_0
        start_net = copy.deepcopy(self)

        # construct the iter point y_t
        iter_net = copy.deepcopy(self)

        # construct the auxiliary point z_1
        iter_net_aux_1 = copy.deepcopy(self)

        # construct the auxiliary point z_2
        iter_net_aux_2 = copy.deepcopy(self)

        # generate random vector y_0 with unit norm
        norm_iter = 0.0
        for param_iter in iter_net.parameters():
            param_iter.data = torch.randn(param_iter.data.size())
            norm_iter += param_iter.data.norm(2) ** 2
        norm_iter = np.sqrt(norm_iter)

        for param_iter in iter_net.parameters():
            param_iter.data /= norm_iter

        if args.cuda:
            iter_net.cuda()

        # estimate_value represents v^{T}Hv
        estimate_value = 0.0
        # SCSG for PCA
        for epoch in range(n_epoch):
            # set estimate_value equal to 0
            estimate_value = 0.0
            # set the inner iteration
            num_data_pass = inner_iteration

            # zero net_aux for sum up
            for param in iter_net_aux_1.parameters():
                param.data = torch.zeros(param.data.size())
            if args.cuda:
                iter_net_aux_1.cuda()
            iter_net_vr = copy.deepcopy(iter_net)

            # calculate the large batch Hessian vector product
            for data_iter, data in enumerate(dataset_power):

                if data_iter > num_data_pass - 1:
                    break

                # get the input and label
                inputs, labels = data

                # wrap data and target into variable
                if args.cuda:
                    input_data, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
                    input_data, labels = Variable(inputs), Variable(labels)

                # zero the gradient
                start_net.zero_grad()

                # get the output
                outputs = start_net.forward(input_data)
                # define the loss for calculating Hessian-vector product
                loss_self_defined = loss_function(outputs, labels)

                # compute the gradient
                grad_params = torch.autograd.grad(loss_self_defined, start_net.parameters(), create_graph=True)
                # compute the Hessian-vector product
                inner_product = 0.0
                for param_vr, param_grad in zip(iter_net_vr.parameters(), grad_params):
                    inner_product += torch.sum(param_vr * param_grad)

                h_v_vr = torch.autograd.grad(inner_product, start_net.parameters(), create_graph=True)

                # sum up the hessian-vector product Hv and lambda * I

                # sum up Hv
                for param_h_v, param_aux_1, param_aux_2 in zip(h_v_vr, iter_net_aux_1.parameters(), iter_net_aux_2.parameters()):
                    param_aux_2 = param_h_v
                    param_aux_1.data -= param_aux_2.data / (num_data_pass * lambda_power)

                # sum up lambda * I
                for param_aux_1, param_iter_vr in zip(iter_net_aux_1.parameters(), iter_net_vr.parameters()):
                    param_aux_1.data += param_iter_vr.data / num_data_pass

            # large-batch term
            iter_net_vr = copy.deepcopy(iter_net_aux_1)
            # sgd term
            iter_net_pre_vr = copy.deepcopy(iter_net)
            # inner iteration
            num_data_pass = inner_iteration

            # inner update
            for data_iter, data in enumerate(dataset_power):

                if data_iter > num_data_pass - 1:
                    break

                # get the input and label
                inputs, labels = data

                # wrap data and target into variable
                if args.cuda:
                    input_data, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
                    input_data, labels = Variable(inputs), Variable(labels)

                # zero the gradient
                start_net.zero_grad()
                outputs = start_net.forward(input_data)

                loss_self_defined = loss_function(outputs, labels)

                # compute the gradient
                grad_params = torch.autograd.grad(loss_self_defined, start_net.parameters(), create_graph=True)

                # compute the Hessian-vector product for current point
                inner_product = 0.0
                for param_iter, param_grad in zip(iter_net.parameters(), grad_params):
                    inner_product += torch.sum(param_iter * param_grad)

                h_v = torch.autograd.grad(inner_product, start_net.parameters(), create_graph=True)

                # compute the Hessian-vector product for previous one
                inner_product_pre = 0.0
                for param_iter_vr, param_grad in zip(iter_net_pre_vr.parameters(), grad_params):
                    inner_product_pre += torch.sum(param_iter_vr * param_grad)

                h_v_pre_vr = torch.autograd.grad(inner_product_pre, start_net.parameters(), create_graph=True)

                # estimate the curvature
                for param_iter, param_h_v in zip(iter_net.parameters(), h_v):
                    estimate_value += torch.sum(param_iter * param_h_v)

                # print every epoch_len mini-batches
                epoch_len = num_data_pass
                if data_iter % epoch_len == epoch_len - 1:
                    estimate_value = float(estimate_value) / (1.0 * epoch_len)
                    print('epoch: %d, estimate_value: %.8f' % (epoch, estimate_value))

                # update SCSG
                norm_iter = 0.0
                # power method
                for param_aux_1, param_h_v, param_h_v_pre, param_iter, param_iter_pre_vr, param_iter_vr in zip(
                        iter_net_aux_1.parameters(), h_v, h_v_pre_vr, iter_net.parameters(), iter_net_pre_vr.parameters(),
                        iter_net_vr.parameters()):
                    param_aux_1 = - param_h_v / lambda_power + param_h_v_pre / lambda_power
                    param_iter.data += learning_rate * (
                        param_aux_1.data + param_iter_vr.data + param_iter.data - param_iter_pre_vr.data)

                    norm_iter += param_iter.data.norm(2) ** 2

                # norm of iter_net
                norm_iter = np.sqrt(norm_iter)

                # normalization for iter_net
                for param_iter in iter_net.parameters():
                    param_iter.data /= norm_iter

        num_data_pass = inner_iteration

        # calculate a large batch gradient for choosing direction for negative curvature
        start_net.zero_grad()
        start_net.calculate_loss_grad(dataset_power, loss_function, num_data_pass)

        # update with negative curvature
        direction_value = 0.0

        # if estimate_value < 0, then take a negative curvature step
        if estimate_value < 0.0:
            for param_start, param_iter in zip(start_net.parameters(), iter_net.parameters()):
                direction_value += torch.dot(param_start.grad, param_iter)

            # update the direction value
            direction_value = float(torch.sign(direction_value))
            # print the direction value
            print('direction_value:', float(direction_value))

            # take a negative curvature step along v
            for param_self, param_iter in zip(self.parameters(), iter_net.parameters()):
                param_self.data -= (direction_value * eta_neg) * param_iter.data
            return estimate_value

# initial the CNN
cnn_net = Net()

if args.cuda:
    cnn_net.cuda()

# use Adam as optimizer
optimizer = optim.Adam(cnn_net.parameters(), lr=args.LR)

# define loss function as Cross Entropy
loss_func = nn.CrossEntropyLoss()

# Training process
for epoch in range(args.EPOCH):

    for step, data in enumerate(trainloader):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn_net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print every args.TRACK_INTERVAL mini-batches
        if step % args.TRACK_INTERVAL == 0:

            # print the training loss and gradient norm for training data
            cnn_net.zero_grad()
            print('Training: ')
            _, full_grad_norm = cnn_net.calculate_loss_grad(trainloader, loss_func, args.SUBSAMPLE_SIZE)

            # print the test loss and gradient norm for test data
            cnn_net.zero_grad()
            print('Test: ')
            cnn_net.calculate_loss_grad(testloader, loss_func, args.SUBSAMPLE_SIZE)

            # if the norm of gradient is small, take negative curvature descent step
            if full_grad_norm < args.NORM_THRESHOLD:
                cnn_net.power_method(train_loader_power, loss_func, args)

