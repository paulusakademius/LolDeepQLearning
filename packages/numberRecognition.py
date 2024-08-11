import torch as T
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def loadData():
    training_dataloader = DataLoader(torchvision.datasets.MNIST('C:/Users/pauld/Desktop/LolRL/data/train',train = True, download = True, transform=torchvision.transforms.ToTensor), shuffle=True)
    test_dataloader = DataLoader(torchvision.datasets.MNIST('C:/Users/pauld/Desktop/LolRL/data/test',train = False, download = True, transform=torchvision.transforms.ToTensor), shuffle=True)
    return training_dataloader, test_dataloader



    

class Network(nn.Module):
    def __init__(self, lr):
        super(Network,self)
        self.conv1 = nn.Conv2d(1, 8 , 4)
        self.pooling = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(8, 8, 4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(392,520)
        self.fc2 = nn.Linear(520, 160)
        self.loss = nn.MSELoss()
        self.optimizer = optim.adam(self.parameters(), lr=lr)

    def forward(self, image):
        x = self.pooling(self.conv1(image))
        x = self.pooling(self.conv2(x))
        x = F.relu(self.fc1(self.flatten(x)))
        x = self.fc2(x)

        return x

    

    

def trainLoop(model,training_dataloader):
    
    for batch, (x,y) in enumerate(training_dataloader):
        size = len(training_dataloader.dataset)
        pred = model.forward(x)
        loss = model.loss(pred,y)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def testLoop(model, test_dataloader):
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with T.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += model.loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(T.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def trainAndSave(epochs):
    model = Network()
    training_dataloader, test_dataloader = loadData()
    for i in range(epochs):
        trainLoop(model,training_dataloader)
        testLoop(model,test_dataloader)
    T.save(model,"C:/Users/pauld/Desktop/LolRL/models/numModel.pth" )


def loadModel():
    model = T.load("C:/Users/pauld/Desktop/LolRL/models/numModel.pth")
    model.eval()
    return model