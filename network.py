import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.set_default_tensor_type(torch.FloatTensor)
IN_FEATURES = 704  # tensors大小，由矩阵大小决定

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


para = {19: 6, 18: 6, 17: 6, 16: 5, 15: 5, 14: 5, 13: 5, 12: 4, 11: 4, 10: 4, 9: 4, 8: 3, 7: 3, 6: 3, 5: 3}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_layer):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(input_layer, 16, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def resnet18(input_layers):
    model = ResNet(block=BasicBlock, layers=[3, 3, 3], input_layer=input_layers)
    return model


class Easy_model(nn.Module):
    def __init__(self, input_layer):
        super(Easy_model, self).__init__()
        self.conv1 = nn.Conv2d(input_layer, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, input):
        input = self.relu(self.bn1(self.conv1(input)))
        input = self.relu(self.bn2(self.conv2(input)))
        input = self.relu(self.bn3(self.conv3(input)))

        return input

# 我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法
class Model(nn.Module):
    def __init__(self, input_layer, row_size, column_size):
        super(Model, self).__init__()
        self.model = Easy_model(input_layer)
        self.p = 6
        self.output_channel = 128
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # value head
        self.value_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=16)
        self.value_bn1 = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(in_features=IN_FEATURES, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)
        # policy head
        self.policy_conv1 = nn.Conv2d(kernel_size=1, in_channels=self.output_channel, out_channels=16)
        self.policy_bn1 = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(in_features=IN_FEATURES, out_features=row_size * column_size)

    # forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。
    def forward(self, state):
        s = self.model(state)
        # value head part
        v = self.value_conv1(s)
        v = self.relu(self.value_bn1(v)).view(-1, IN_FEATURES)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p)).view(-1, IN_FEATURES)
        prob = self.policy_fc1(p)
        return prob, value


class neuralnetwork:
    def __init__(self, input_layers, row_size, column_size, use_cuda=False, learning_rate=0.1):
        # 配置GPU
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = Model(input_layer=input_layers, row_size=row_size, column_size=column_size).cuda().float()
        else:
            self.model = Model(input_layer=input_layers, row_size=row_size, column_size=column_size).float()
        # 使用SGD随机梯度下降 优化
        self.opt = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)
        self.mse = nn.MSELoss()  # 均方误差
        self.crossloss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def train(self, data_loader, game_time):
        self.model.train()  # model.train()的作用是启用 Batch Normalization 和 Dropout
        loss_record = []
        for batch_idx, (state, distrib, winner) in enumerate(data_loader):
            tmp = []
            # Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息
            # 3个参数对应训练数据的state,distribution,value
            # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。把winner变成列向量
            state, distrib, winner = Variable(state).float(), Variable(distrib).float(), Variable(winner).unsqueeze(1).float()
            if self.use_cuda:
                state, distrib, winner = state.cuda(), distrib.cuda(), winner.cuda()

            self.opt.zero_grad()  # optimizer.zero_grad()的作用是清除所有可训练的torch.Tensor的梯度
            prob, value = self.model(state)
            output = F.log_softmax(prob, dim=1)  # 按照行和列做归一化后再做一次log操作
            cross_entropy = - torch.mean(torch.sum(distrib * output, 1))  # 交叉熵
            cross_entropy = cross_entropy.to(torch.float32)
            mse = F.mse_loss(value, winner)  # 均方误差
            mse = mse.to(torch.float32)
            loss = cross_entropy + mse  # 损失函数loss定义了模型优劣的标准，loss越小，模型越好
            loss.backward()  # 将loss向输入测进行反向传播计算梯度
            self.opt.step()  # 更新参数

            tmp.append(cross_entropy.data)
            if batch_idx % 10 == 0:
                print("We have played {} games, and batch {}, the cross entropy loss is {}, the mse loss is {}".format(
                    game_time, batch_idx, cross_entropy.data, mse.data))
                loss_record.append(sum(tmp) / len(tmp))
        return loss_record

    # 在使用model.eval()时就是将模型切换到测试模式，
    # 在这里，模型就不会像在训练模式下一样去更新权重。但是需要注意的是model.eval()不会影响各层的梯度计算行为，
    # 即会和训练模式一样进行梯度计算和存储，只是不进行反向传播。
    def eval(self, state):  # state大小有问题?
        self.model.eval().float()
        # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
        # 在eval模式下，Dropout层会让所有的激活单元都通过，
        # 而Batch Normalization层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
        self.use_cuda = False
        if self.use_cuda:
            state = torch.from_numpy(state).unsqueeze(0).float().cuda()
        else:
            state = torch.from_numpy(state).unsqueeze(0).float()
        with torch.no_grad():  # torch.no_grad()用于停止autograd的计算，能起到加速和节省显存的作用
            prob, value = self.model(state)
        return F.softmax(prob, dim=1), value

    def adjust_lr(self, lr):  # 调整学习率
        for group in self.opt.param_groups:
            group['lr'] = lr
