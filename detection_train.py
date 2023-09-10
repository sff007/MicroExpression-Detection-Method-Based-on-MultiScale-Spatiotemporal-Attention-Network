#!user/bin/python
import os
import numpy as np
import torch

from torch.optim import lr_scheduler
from data.dataload import VideoDataset
from models.mobilev2_3d import get_model
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
from tensorboardX import SummaryWriter
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
def train_model(acc_list,i,tr_dataset_sizes,te_dataset_sizes,train_data_loaders,test_data_loaders,model, criterion, optimizer, scheduler, num_epochs):
    total_step =len(train_dataloader)

    writer = SummaryWriter('./result/xiaorong/multi_scale/log/'+str(i))
    full_path1 = './result/xiaorong/multi_scale/test1.txt'
    full_path2 = './result/xiaorong/multi_scale/test2.txt'
    f1 = open(full_path1,'a')
    f2 = open(full_path2, 'a')
    since = time.time()
    num = 0
    best_acc = 0.0
    best_epoch = 0
    end_running_corrects=0
    iter_epoch =0
    f1.write('sub'+str(i))
    f1.write('\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        f1.write('epoch            '+str(epoch))
        f1.write('\n')
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer.step()
                scheduler.step()

                model.train(True)
                running_loss = 0.0
                running_corrects = 0

                for data in train_data_loaders:
                    inputs, labels = data
                    if use_gpu:
                        inputs =inputs.to(device)
                        labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()* inputs.size(0)
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / tr_dataset_sizes
                writer.add_scalar('train/loss', epoch_loss, epoch)
                epoch_acc = running_corrects / tr_dataset_sizes
                writer.add_scalar('train/acc', epoch_acc, epoch)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                f1.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                f1.write('\n')

            else:
                model.train(False)
                running_loss = 0.0
                running_corrects = 0

                d=1
                for data in test_data_loaders:


                    inputs, labels = data



                    if use_gpu:
                        inputs =inputs.to(device)
                        labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs= model(inputs)

                    _, preds = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += float(torch.sum(preds == labels.data))
                    d=d+1

                epoch_loss = running_loss / te_dataset_sizes
                writer.add_scalar('test/loss', epoch_loss, epoch)
                epoch_acc = running_corrects /te_dataset_sizes
                writer.add_scalar('test/acc', epoch_acc, epoch)
                acc_list.append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                f1.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                f1.write('\n')



            if phase == 'val' and epoch_acc > best_acc:
                iter_epoch = epoch
                best_epoch=epoch
                m=model

                end_running_corrects = running_corrects

                best_acc = epoch_acc
                num = running_corrects
                filename = 'best_epoch{}_model.pt'.format(epoch)
                save_path = './result/xiaorong/multi_scale/model' + '/' + 'sub' + str(i) +'/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                s_p = save_path+filename
                print('s_p',s_p)
                if (os.path.exists(s_p)) is False:
                    torch.save(m.state_dict(),s_p)
                acc_list.append(num)

        print('best_acc', best_acc)
        print('best_epoch',best_epoch)
        print('running_corrects',num)
        if best_acc == 1:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc epoch: {:4f}'.format(iter_epoch))
    #f.write('....................................')
    f2.write('sub' + str(i))
    f2.write('\n')
    f2.write('best.{},Loss: {:.4f} Acc: {:.4f}'.format(iter_epoch, epoch_loss, best_acc))
    f2.write('<<<<<<<<<<correct_numbers>>>>>>>' + str(end_running_corrects))
    f2.write('\n')
    return model
if __name__ =='__main__':
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    acc_list = []
    for aa in range(1,6):
        model = get_model(num_classes=2, width_mult=1.).to(device)
        print('----------------here-------------sub------------', aa)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        video_root = "G:\\MER\\CASME2_10_spotting_crop"
        train_video_list = "G:\\MER\\detect\\train_" + str(aa) + ".txt"
        test_video_list = "G:\\MER\\detect\\test_"+str(aa)+".txt"
        train_dataloader = VideoDataset(video_root,train_video_list, transform=transforms)
        test_dataloader = VideoDataset(video_root,test_video_list, transform=transforms)
        tr_dataset_sizes = len(train_dataloader)
        te_dataset_sizes = len(test_dataloader)
        train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=1,
                                                   shuffle=False )
        test_loader = torch.utils.data.DataLoader(test_dataloader, batch_size=1,
                                                  shuffle=False)
        mm = train_model(acc_list,aa,tr_dataset_sizes, te_dataset_sizes, train_loader, test_loader, model, criterion, optimizer,
                         scheduler, 100)


