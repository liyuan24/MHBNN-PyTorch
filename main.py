import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from custom_dataset import *
from mhbn import *
import time
import util
import argparse


transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



#cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='MHBNN-PyTorch')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-nlocal' ,'--n-local-features', default=10, type=int, metavar='N', help='number of local features(default: 10)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()


#root path of data
root = args.data


#the batch size of train and validation
batch_size = args.batch_size


# Load dataset
dset_train = MultiViewDataSet(root, 'train', transform=transform)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=2)

dset_val = MultiViewDataSet(root, 'test', transform=transform)
val_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=True, num_workers=2)


classes, class_to_idx = dset_train.classes, dset_train.class_to_idx


#number of local features
num_local_features = args.n_local_features


model = mhbnn(True, len(classes), num_local_features)
model.to(device)
cudnn.benchmark = True
print('Running on ' + str(device))


lr = args.lr
lr_decay = args.lr_decay
lr_decay_freq = args.lr_decay_freq
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0
print_freq = args.print_freq


resume_path = './checkpoint'
resume_ckp_path = args.resume



def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(resume_ckp_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(resume_ckp_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



def train():
    train_size = len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        #convert to shape [batch_size, views, channels, width, height]
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) %  print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))


def eval(data_loader, is_test=False):
    if is_test:
       load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()
    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


if resume_ckp_path:
    load_checkpoint()



for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    model.train()
    train()
    print('Time taken: %.2f sec.' % (time.time() - start))

    model.eval()
    avg_test_acc, avg_loss = eval(val_loader)

    print('\nEvaluation:')
    print('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
    print('\tCurrent best  val acc: %.2f' % best_acc)
    

    # Log epoch to tensorboard
    # See log using: tensorboard --logdir='logs' --port=6006
#     util.logEpoch(logger, model, epoch + 1, avg_loss, avg_test_acc)

    # Save model
    if avg_test_acc > best_acc:
        print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
        best_acc = avg_test_acc
        best_loss = avg_loss
        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': avg_test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, 'mhbnn_'+str(num_local_features))

    # Decaying Learning Rate
    if (epoch + 1) % lr_decay_freq == 0:
        lr *= lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('Learning rate:', lr)

