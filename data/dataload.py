import os, sys, shutil,re
from PIL import Image
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

def load_images_triple(video_root, video_list):
    images = list()
    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):
            video= line.strip().split()
            video_name = video[0]  # name of video
            label = video[1]  # label of video
            video_path = os.path.join(video_root, video_name)
            img_lists = os.listdir(video_path)
            img_lists.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            img_count = len(img_lists)
            num_per_part = int(img_count) // 4
            if int(img_count) > 4:
                for i in range(1):
                    random_select_first = random.randint(0, num_per_part)
                    random_select_second = random.randint(num_per_part, num_per_part * 2)
                    random_select_third = random.randint(num_per_part * 2, num_per_part * 3)
                    random_select_forth = random.randint(3 * num_per_part, len(img_lists) - 1)
                    img_path_first = os.path.join(video_path, img_lists[random_select_first])
                    img_path_second = os.path.join(video_path, img_lists[random_select_second])
                    img_path_third = os.path.join(video_path, img_lists[random_select_third])
                    img_path_forth = os.path.join(video_path, img_lists[random_select_forth])

                    images.append((img_path_first, label))
                    images.append((img_path_second, label))
                    images.append((img_path_third, label))
                    images.append((img_path_forth, label))
            else:
                for j in range(1):
                    img_path_first = os.path.join(video_path, img_lists[j])
                    img_path_second = os.path.join(video_path, random.choice(img_lists))
                    img_path_third = os.path.join(video_path, random.choice(img_lists))
                    img_path_forth = os.path.join(video_path, random.choice(img_lists))

                    images.append((img_path_first, label))
                    images.append((img_path_second, label))
                    images.append((img_path_third, label))
                    images.append((img_path_forth, label))
            index.append(np.ones(img_count) * id)
        index = np.concatenate(index, axis=0)
        index = index.astype(int)
    return images, index


class TripleImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, transform=None):

        self.images, self.index = load_images_triple(video_root, video_list)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.images[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            target = int(target)
        #     torch.FloatTensor(target)
        return image, target, self.index[index]

    def __len__(self):
        # 数据集长度
        return len(self.images)


def load_imgs_total_frame(video_root, video_list):

    imgs = list()
    with open(video_list, 'r') as imf:
        index = []
        video_names = []
        for id, line in enumerate(imf):
            video_label = line.strip().split()
            video_name = video_label[0]
            label = video_label[1]
            video_path = os.path.join(video_root, video_name)
            index.append(id)
            imgs.append((video_path,label))
    print(imgs)
    return imgs, index

class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, transform=None):

        self.imgs, self.index = load_imgs_total_frame(video_root, video_list)
        self.transform = transform

    def __getitem__(self, index):
        data_jpgs_list = []
        path, target = self.imgs[index]
        img_lists = os.listdir(path)
        img_lists.sort(key=lambda x:int(x[:-4]))
        for j in range (len(img_lists)):
            data =Image.open(path+'//'+img_lists[j])
            data = self.transform(data)
            data = data.view(3,192,160)
            data_jpgs_list.append(data)
        data_jpgs = torch.stack(data_jpgs_list,0)
        return data_jpgs,int(target)
    def __len__(self):
        return len(self.imgs)


def Load_data(root, list_train, list_eval, list_test, batchsize):

    train_dataset = TripleImageDataset(
        video_root=root,
        video_list=list_train,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    val_dataset = VideoDataset(
        video_root=root,
        video_list=list_eval,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_dataset = TripleImageDataset(
        video_root=root,
        video_list=list_test,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)

    dev_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=0, pin_memory=True)

    return train_loader, dev_loader, test_loader

