import torch
from torch import nn
from torch import optim

def init_model(indim, outdim):
    model = nn.Sequential()


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(156, 86)
        self.hidden2 = nn.Linear(86, 16)
        self.output = nn.Linear(16, 2)
        #self.out_tensor = self.output(input)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, inputvect):
        layer1 = self.hidden1(inputvect)
        layer1relu = self.relu(layer1)
        layer2 = self.hidden2(layer1relu)
        layer2relu = self.relu(layer2)
        outlayer = self.output(layer2relu)
        output = self.softmax(outlayer)
        return output

    """
    def learn(self, target, learning_rate, max_iterations):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        for i in range(max_iterations):
            optimizer.zero_grad()
            self.out_tensor = self(input)
            loss = criterion(self.out_tensor, target)
            loss.backward()
            optimizer.step()
    """



