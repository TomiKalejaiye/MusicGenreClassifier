#%%
#===================================================================
# Genre Classification Using CNN
#(c) 2019   ok93@cornell.edu, bjy26@cornell.edu
#===================================================================

import numpy as np 
import scipy as sp
from scipy.io import loadmat 
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
#%%
#Compute mel spectrograms from STFTs
song_mels = []
song_labels = []
for f in glob.glob("stdft_data_gtzam/*.mat"):
    #Read in the data
    song=loadmat(f)
    song_stft = song['magnitude']

    song_stft_truncated = np.array(song_stft[:126,:])

    mel = librosa.feature.melspectrogram(S=song_stft_truncated.T, n_mels=128,center=False)

    song_mels.append(mel[np.newaxis,np.newaxis,...])
    song_labels.append(f.split('/')[1][:-4])
#%%
#Map genre labels to index numbers
import re
for i,label in enumerate(song_labels):
    song_labels[i] = re.sub(r'\d+', '', label)

#Replace word labels with numbers
genres = ['rock','reggae','pop','metal','jazz','hiphop','disco','country','classical','blues']
song_labels = np.array(song_labels)

song_labels = np.array([genres.index(i) for i in song_labels])

song_mels = np.concatenate(song_mels,axis=0)

#%%
#Save data and labels so we don't have to repeat previous steps every time.
np.save('song_mels',song_mels)
np.save('song_labels',song_labels)
#%%
song_mels = np.load('song_mels.npy')
song_labels = np.load('song_labels.npy')
#%%
#Create test and train loaders.
batch_size = 32

num_train = 30438

rnd_idx = np.array(range(song_mels.shape[0]))
np.random.shuffle(rnd_idx)

train_data = torch.utils.data.TensorDataset(torch.from_numpy(song_mels[rnd_idx[:num_train]]), torch.from_numpy(song_labels[rnd_idx[:num_train]]))
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = torch.utils.data.TensorDataset(torch.from_numpy(song_mels[rnd_idx[num_train:]]), torch.from_numpy(song_labels[rnd_idx[num_train:]]))
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
#%%
#Import and train model
from myCNNs import *
import torch.optim as optim

import shutil

def save_ckp(state):
    f_path = 'checkpoint2.pth'
    torch.save(state, f_path)

model = musicCNNDeep().double()

num_epochs = 150

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_over_epochs = []
loss_every_nbatches = []
acc_every_nbatches = []
avg_loss_every_nbatches = []

model.train()

for epoch in range(num_epochs): 
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
    
        optimizer.zero_grad()
        inputs, labels = data
        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs = model(inputs)
            _,preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'batch': i
                    }
        
        save_ckp(checkpoint)

        # print statistics
        running_loss += loss.item()
        if i % 50 ==49:    # print every n mini-batches
            loss_every_nbatches.append(loss.item())
            acc_every_nbatches.append(correct/total)
            print('[%d, %5d] loss: %.3f acc: %.2f' %
                  (epoch + 1, i + 1, running_loss / 50,(correct/total)))
            avg_loss_every_nbatches.append(running_loss / 50)
            running_loss = 0.0
    loss_over_epochs.append(loss.item())

print('Finished Training')

#%%
#Plot results from training.
loss = np.load('avg_loss_every_nbatches.npy')

plt.plot(50*np.arange(len(loss)),loss,label='training loss')
plt.legend()
plt.title('Training Loss Every 50 Batches')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.savefig('LossPlot')

acc = np.load('acc_every_nbatches.npy')
plt.plot(50*np.arange(len(acc)),acc,label='training accuracy')
plt.legend()
plt.title('Training Accuracy Every 50 Batches')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.savefig('AccPlot')

# %%
#Calculate accuracy on test set.
correct = 0
total = 0
all_pred = []
all_labels = []
with torch.no_grad():
    for i,data in enumerate(test_loader):
        inputs,labels = data
        outputs = model(inputs.double())
        _, predicted = torch.max(outputs.data, 1)
        all_pred.append(predicted.numpy())
        all_labels.append(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test examples: %d %%' % (total,
    100 * correct / total))

#%%
#Generate confusion matrix for classifications
all_labels = np.concatenate(all_labels)
all_pred = np.concatenate(all_pred)

genres = ['rock','reggae','pop','metal','jazz','hiphop','disco','country','classical','blues']

all_labels_txt = [genres[int(i)] for i in all_labels]
all_pred_txt = [genres[int(i)] for i in all_pred]

confusion = confusion_matrix(all_labels_txt, all_pred_txt, normalize='true')
confusion_df = pd.DataFrame(confusion)

plt.figure()
plt.title('Genre Classifier Confusion Matrix')
sn.heatmap(confusion_df, annot=True, fmt='.2f', xticklabels=genres, yticklabels=genres)
plt.savefig('ConfusionMatrix')
# %%
