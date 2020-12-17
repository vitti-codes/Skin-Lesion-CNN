from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from Augmented_HAM10000_dataset import Augmented_HAM10000_Dataset


class ConvNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes=7):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        #this is called a same convolution, n_in == n_out, the dimension of the image will not change in convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(in_features=16 * 16*16, out_features=num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) #flattening
        x = self.fc1(x)

        return x

#GPU support for parallel computing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
num_classes = 7
learning_rate = 0.001
batch_size = 64
epochs = 500
in_channels = 3

#load data (images with labels)
'''directory = 'C:/Users/vitti/PycharmProjects/task2_mnist_702cw/'
df = pd.read_csv(directory + 'augmented_dataframe.csv')

file_paths = df['image_id'].values
print(file_paths.shape)
labels = df['target'].values

ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)'''

#loading in our augmented HAM10000 dataset, and splitting it into a training and testing set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dir = 'C:/Users/vitti/PycharmProjects/task2_mnist_702cw/Augmented_images/'
dataset = Augmented_HAM10000_Dataset(csv_file='augmented_data_2.csv', root_dir=dir, transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [35167, 11723])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)



#train_loader, test_loader = train_test_split(X, y, random_state=7, shuffle=True)

#initializing our convolutional neural network
model = ConvNet().to(device)
#x = torch.randn(64,3, 64, 64)
#print(x.shape)
#exit()
#Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
n_iter = 0

#training our CNN
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        output = model(images)
        loss = criterion(output, labels)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_acc(loader, model):
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)

        print(f"Got {num_correct} / {num_samples} "
              f" with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
check_acc(train_loader,model)
model.eval()
check_acc(train_loader, model)
model.train()


PATH = "cnn_model_final.pt" #specifying the path to save model
torch.save(model, PATH) #saving model
cnn_model = torch.load(PATH)
cnn_model.eval()