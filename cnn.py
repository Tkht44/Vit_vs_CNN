import torch.nn as nn


class Net(nn.Module):
    def __init__(self, class_num, conv_size, linear_size):
        super(Net, self).__init__()
        self.conv_size = conv_size
        self.conv1 = nn.Conv2d(conv_size[0], conv_size[1], 5)
        self.pool = nn.MaxPool2d(2, 2)  # 半分にする
        self.conv2 = nn.Conv2d(conv_size[1], conv_size[2], 5)
        self.fc1 = nn.Linear(conv_size[2] * 5 * 5, linear_size[0])
        self.fc2 = nn.Linear(linear_size[0], linear_size[1])
        self.fc3 = nn.Linear(linear_size[1], class_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.conv_size[2] * 5 * 5)  # tensorを行列に
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
