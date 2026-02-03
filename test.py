import torch
import torchvision
from torchvision import transforms, datasets, models
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# other
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

batch_size=200
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.MNIST(root = './dataset/02/',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = datasets.MNIST(root = './dataset/02/',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

torch.cuda.empty_cache()

resnet18_pt = models.resnet18(pretrained=True).to(device)
resnet34_pt = models.resnet34(pretrained=True).to(device)

# MNIST setting
def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 10, bias=True)
    return model

change_layers(resnet18_pt).to(device)
change_layers(resnet34_pt).to(device)

criterion = nn.CrossEntropyLoss()
error_fn = nn.MSELoss()
optimizer_rn18 = optim.SGD(resnet18_pt.parameters(), lr=0.001,
                      momentum=0.9)
optimizer_rn34 = optim.SGD(resnet34_pt.parameters(), lr=0.001,
                      momentum=0.9)

# Training
def train(epoch, model, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = train_loss/total
    epoch_acc = correct/total*100
    print("Train | Loss:%.4f" 
        % epoch_loss)
    return epoch_loss, epoch_acc

# Training
def ft_train(epoch, model1, model2, criterion, error_fn, optimizer1, optimizer2):
    model1.train()
    model2.train()
    train_loss = 0
    correct1 = 0
    correct2 = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        loss = (0.5 * criterion(outputs1, labels) + 0.5 * criterion(outputs2, labels)) + error_fn(outputs1, outputs2) # setting
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        train_loss += loss.item()*inputs.size(0)
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        total += labels.size(0)
        correct1 += predicted1.eq(labels).sum().item()
        correct2 += predicted2.eq(labels).sum().item()
    
    epoch_loss = train_loss/total
    epoch_acc1 = correct1/total*100
    epoch_acc2 = correct2/total*100
    print("F-Train | Loss:%.4f " 
        % epoch_loss)
    return epoch_loss, epoch_acc1, epoch_acc2

def test(epoch, model, criterion, optimizer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()*inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = test_loss/total
        epoch_acc = correct/total*100
        print("Test | Loss:%.4f" 
            % epoch_loss)
    return epoch_loss, epoch_acc

# train ResNet18
start_time = time.time()
#best_acc = 0
epoch_length = 50
save_loss_rn18 = {"train":[],
             "test":[]}
save_acc_rn18 = {"train":[],
             "test":[]}
for epoch in range(epoch_length):
    print("Epoch %s" % epoch)
    train_loss, train_acc = train(epoch, resnet18_pt, criterion, optimizer_rn18)
    save_loss_rn18['train'].append(train_loss)
    save_acc_rn18['train'].append(train_acc)

    test_loss, test_acc = test(epoch, resnet18_pt, criterion, optimizer_rn18)
    save_loss_rn18['test'].append(test_loss)
    save_acc_rn18['test'].append(test_acc)

    # Save model
    #if test_acc > best_acc:
    #    best_acc = test_acc
    #    best_model_wts = copy.deepcopy(resnet_pt.state_dict())
    #resnet_pt.load_state_dict(best_model_wts)

learning_time = time.time() - start_time
print(f'**Learning time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s')

# train ResNet34
start_time = time.time()
#best_acc = 0
epoch_length = 50
save_loss_rn34 = {"train":[],
             "test":[]}
save_acc_rn34 = {"train":[],
             "test":[]}
for epoch in range(epoch_length):
    print("Epoch %s" % epoch)
    train_loss, train_acc = train(epoch, resnet34_pt, criterion, optimizer_rn34)
    save_loss_rn34['train'].append(train_loss)
    save_acc_rn34['train'].append(train_acc)

    test_loss, test_acc = test(epoch, resnet34_pt, criterion, optimizer_rn34)
    save_loss_rn34['test'].append(test_loss)
    save_acc_rn34['test'].append(test_acc)

    # Save model
    #if test_acc > best_acc:
    #    best_acc = test_acc
    #    best_model_wts = copy.deepcopy(resnet_pt.state_dict())
    #alexnet_pt.load_state_dict(best_model_wts)

learning_time = time.time() - start_time
print(f'**Learning time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s')

torch.cuda.empty_cache()

resnet18_pt = models.resnet18(pretrained=True).to(device)
resnet34_pt = models.resnet34(pretrained=True).to(device)

change_layers(resnet18_pt).to(device)
change_layers(resnet34_pt).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_rn18 = optim.SGD(resnet18_pt.parameters(), lr=0.001,
                      momentum=0.9)
optimizer_rn34 = optim.SGD(resnet34_pt.parameters(), lr=0.001,
                      momentum=0.9)

# train R-R
start_time = time.time()
#best_acc = 0
epoch_length = 50
save_loss_rr18 = {"train":[],
             "test":[]}
save_loss_rr34 = {"train":[],
             "test":[]}
save_acc_rr18 = {"train":[],
             "test":[]}
save_acc_rr34 = {"train":[],
             "test":[]}
for epoch in range(epoch_length):
    print("Epoch %s" % epoch)
    train_loss, train_acc1, train_acc2 = ft_train(epoch, resnet18_pt, resnet34_pt, criterion, error_fn, optimizer_rn18, optimizer_rn34)
    save_loss_rr18['train'].append(train_loss)
    save_loss_rr34['train'].append(train_loss)
    save_acc_rr18['train'].append(train_acc1)
    save_acc_rr34['train'].append(train_acc2)

    test_loss, test_acc = test(epoch, resnet18_pt, criterion, optimizer_rn18)
    save_loss_rr18['test'].append(test_loss)
    save_acc_rr18['test'].append(test_acc)
    test_loss, test_acc = test(epoch, resnet34_pt, criterion, optimizer_rn34)
    save_loss_rr34['test'].append(test_loss)
    save_acc_rr34['test'].append(test_acc)

    # Save model
    #if test_acc > best_acc:
    #    best_acc = test_acc
    #    best_model_wts = copy.deepcopy(resnet_pt.state_dict())
    #resnet_pt.load_state_dict(best_model_wts)

learning_time = time.time() - start_time
print(f'**Learning time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s')

plt.subplot(2, 1, 1)
plt.plot(range(epoch_length), save_loss_rn18["train"])
plt.plot(range(epoch_length), save_loss_rn34["train"])
plt.plot(range(epoch_length), save_loss_rr18["train"])
plt.plot(range(epoch_length), save_loss_rr34["train"])
plt.legend(('train_loss_rn18', 'train_loss_rn34', 'train_loss_rr18', 'train_loss_rr34'))
plt.subplot(2, 1, 2)
plt.plot(range(epoch_length)[10:], save_loss_rn18["test"][10:])
plt.plot(range(epoch_length)[10:], save_loss_rn34["test"][10:])
plt.plot(range(epoch_length)[10:], save_loss_rr18["test"][10:])
plt.plot(range(epoch_length)[10:], save_loss_rr34["test"][10:])
plt.legend(('test_loss_rn18', 'test_loss_rn34', 'test_loss_rr18', 'test_loss_rr34'))
plt.savefig()

plt.subplot(2, 1, 1)
plt.plot(range(epoch_length), save_acc_rn18["train"])
plt.plot(range(epoch_length), save_acc_rn34["train"])
plt.plot(range(epoch_length), save_acc_rr18["train"])
plt.plot(range(epoch_length), save_acc_rr34["train"])
plt.legend(('train_acc_rn18', 'train_acc_rn34', 'train_acc_rr18', 'train_acc_rr34'))
plt.subplot(2, 1, 2)
plt.plot(range(epoch_length)[10:], save_acc_rn18["test"][10:])
plt.plot(range(epoch_length)[10:], save_acc_rn34["test"][10:])
plt.plot(range(epoch_length)[10:], save_acc_rr18["test"][10:])
plt.plot(range(epoch_length)[10:], save_acc_rr34["test"][10:])
plt.legend(('test_acc_rn18', 'test_acc_rn34', 'test_acc_rr18', 'test_acc_rr34'))
plt.savefig()
