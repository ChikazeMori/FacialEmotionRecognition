import create_datasets as cd
import time, os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image


# to avoid Error #15: Initializing libiomp5.dylib on Mac OS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(model='CNN',rep=5):

    # define a CNN model
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16*22*22, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 8)
            # Define proportion or neurons to dropout
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*22*22)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # choose a model to train
    if model == 'CNN':
        net = Net()
    elif model == 'AlexNet':
        net = torchvision.models.alexnet()
    elif model == 'ResNet18':
        net = torchvision.models.resnet18()
    elif model == 'ResNet50':
        net = torchvision.models.resnet50()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # datasets
    train_label_list = cd.create_datasets(type='train')
    
    os.chdir('..')

    train = []
    for i,row in train_label_list.iterrows():
        img_path = 'CW_Dataset/' + 'train' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0])+'_aligned.jpg'
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)
        train.append([img,label])
    trainLoader = torch.utils.data.DataLoader(train, shuffle=True)

    # training
    start_time = time.time()
    rep = rep # number of epochs (5 or 20 were selected for the report)
    for epoch in range(rep):  # loop over the dataset multiple times

        running_loss = 0.0
        for i,data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        scheduler.step()

    end_time = time.time() - start_time
    print('Finished Training ', end_time)

    # save the model
    os.chdir('..')
    torch.save(net,'Models/' + model + '_' + str(rep) + '.p')
    pickle.dump(end_time, open('Code/' + model + '_' + str(rep) + '_speed.p', 'wb'))
    os.chdir('Code')
    
def test(model='CNN', rep=5):
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 5)
            self.conv2 = nn.Conv2d(4, 8, 5)
            self.conv3 = nn.Conv2d(8, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16*9*9, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            # Define proportion or neurons to dropout
            self.dropout = nn.Dropout(0.1)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 16*9*9)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    model_name = model + '_' + str(rep)
    os.chdir('..')
    
    if model == 'CNN':
        net = Net()
        net.load_state_dict(torch.load('Models/' + model_name + '.p', map_location=torch.device('cpu')))
    else:
        net = torch.load('Models/' + model_name + '.p', map_location=torch.device('cpu'))


    # datasets
    os.chdir('Code')
    test_label_list = cd.create_datasets(type='test')
    os.chdir('..')
    test = []
    for i,row in test_label_list.iterrows():
        img_path = 'CW_Dataset/' + 'test' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0])+'_aligned.jpg'
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)
        test.append([img,label])
    testLoader = torch.utils.data.DataLoader(test, shuffle=True)

    # test
    correct = 0
    total = 0
    classes = (1, 2, 3, 4, 5, 6, 7)
    class_correct = list(0 for i in range(7))
    fp = list(0 for i in range(7))
    tn = list(0 for i in range(7))
    fn = list(0 for i in range(7))
    class_total = list(0 for i in range(7))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            label = labels - 1
            if predicted == labels:
                class_correct[label] += 1
            l = labels
            p = predicted
            if p != l:
                fn[label] += 1
            for i in range(7):
                i += 1
                if i != l:
                    if i == p:
                        fp[i-1] += 1
                    elif i != p:
                        tn[i-1] += 1
            class_total[label] += 1

    print('Accuracy of the network on the testset: %d %%' % (
            100 * correct / total))
    for i in range(7):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
            
    recall = []
    precision = []
    f1 = []
    tp = class_correct
    for i in range(7):
        r  = tp[i]/(tp[i]+fn[i])
        pre = tp[i]/(tp[i]+fp[i])
        f = 2*(r*pre)/(r+pre)
        recall.append(round(r,2))
        precision.append(round(pre,2))
        f1.append(round(f,2))
    
    print(len(recall))
    print('recall',recall)
    print('precision',precision)
    print('f1',f1)