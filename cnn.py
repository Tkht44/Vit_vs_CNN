import torch.nn as nn

class_num = 10
# set hyper param
conv_1 = 3
conv_2 = 6
conv_3 = 16
fc1 = 120
fc2 = 84


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(conv_1, conv_2, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 半分にする
        self.conv2 = nn.Conv2d(conv_2, conv_3, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size([0], -1))  # tensorを行列に
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


net = CNN()
