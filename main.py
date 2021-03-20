import torch
from torch import nn
from pathlib import Path
import wikipedia
from preprocessing import *
from NeuralNet import *
import unidecode
import csv
import random

if __name__ == '__main__':
    #create_csv('english_words.txt', 'english_vectors.csv')

    english_tensors = create_tensors('english_vectors.csv')
    viet_tensors = create_tensors('viet_vectors.csv')
    random.shuffle(english_tensors)
    random.shuffle(viet_tensors)
    tensorlist = viet_tensors + english_tensors
    keylist = list(zip(list(range(6000)), tensorlist))
    random.shuffle(keylist)
    NN = NeuralNet()
    optimizer = optim.SGD(NN.parameters(), lr=0.001)
    for i in keylist:
        optimizer.zero_grad()
        output = NN.forward(i[1])
        if i[0] <= 3000:
            target = torch.tensor([1.,0.])
        else:
            target = torch.tensor([0.,1.])
        criterion = nn.MSELoss()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        print(NN.forward(i[1]))
    print(NN)
    #for i in keylist:
    #    print(NN.forward(i[1]))






