from torch import nn


class SimplerMLP(nn.Module):
    def __init__(self, input_size=33):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ComplexMLP(nn.Module):
    def __init__(self, input_size=33):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        return out


class SimplerLocationModel(SimplerMLP):
    def __init__(self):
        super().__init__(input_size=33)


class ComplexLocationModel(ComplexMLP):
    def __init__(self):
        super().__init__(input_size=33)

class ComplexLocationTimeModel(ComplexMLP):
    def __init__(self):
        super().__init__(input_size=43)

class ComplexLocationTimeSpeedModel(ComplexMLP):
    def __init__(self):
        super().__init__(input_size=44)