import sys
import torch
from time import time
import argparse
from util import dataset, transform, config
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from util.util import check_makedirs
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.optim as optim
import datetime
import math
import os
import pytorch_iou
import cv2
from model import SZ224_Full_Res as basicnet





#TODO: Training and Testing Setup (Please modify parameters here for training or testing)
Flag_train_test = 1             #0 for train and 1 for test.
train_device = 'cuda:0'
test_device  = 'cuda:0'
save_num = 1                    #The number of epochs between which to save the model.
if Flag_train_test==1:          #The time ID of the corresponding training task.
    modelDATE = "2023-03-27-05:43:07.785290"
test_rangeL=20                  #The start and end epochs of the test model.
test_rangeR=60

#Use this command in the console for training: nohup python main.py > test1.log &





print('torch.cuda is available:',torch.cuda.is_available())
print('python version:',sys.version)
print('torch version:',torch.__version__)

eps = math.exp(-10)



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cod_mgl50.yaml', help='config file')
    parser.add_argument('opts', help='see config/cod_mgl50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg



def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x



class MyLossEdge:
    def __init__(self):
        self.x = None
        self.y = None
        self.IOU = pytorch_iou.IOU(size_average=True)

    def loss(self, X, y):  # Q is B node_num n
        lossAll = 0
        for x in X:
            loss = (-y.mul(torch.log(x + eps)) - (1 - y).mul(torch.log(1 - x + eps))).sum() + (abs(x-y)).sum()
            num_pixel = y.numel()
            lossAll = lossAll + torch.div(loss, num_pixel) + self.IOU(x, y)

        return lossAll



def loss_curve(counter, losses):
    fig = plt.figure()
    plt.plot(counter, losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number')
    plt.ylabel('loss')
    plt.show()



def train(loss_fn, args):
    train_losses = []
    train_counter = []
    tnum=0
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_data = dataset.SemData(split='train', data_root=None, data_list=args.train_list,
                                 transform=train_transform)
    train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    ##define the graph model, loss function, optimizer.
    net = basicnet.AESINet(args.block_num, args.block_nod, args.img_dim, args.cov_loop, args.cov_bias, res_pretrained=True)
    device = torch.device(train_device if torch.cuda.is_available() else 'cpu')
    model = net.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0000,
                           amsgrad=False)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    date_str = str(datetime.datetime.now().date())+'-'+str(datetime.datetime.now().time())  #now time
    print("***********",date_str,"**********")
    model_num = 0
    model.train()
    lossAvg = 0
    
    for epoch in range(model_num,args.epoch_num):
        for i, (input, gt, edgegt, _ ) in enumerate(train_loader):
            input =input.to(device)
            gt = gt.to(device)
            edgegt = edgegt.to(device)
            optimizer.zero_grad()
            s1,s2,s3,s4,s5= model(input)
            Out = [s1,s2,s3,s4,s5]
            gt = MaxMinNormalization(gt)
            loss = loss_fn.loss(Out,gt)
            loss.backward()
            optimizer.step()
            lossAvg = lossAvg + loss.item()
            tnum=tnum+1
            train_losses.append(loss.item())
            train_counter.append(((epoch) * len(train_data) / args.train_batch_size) + i)

            if (i + 1 == len(train_loader)):
                print('Train Epoch: {} [{:.0f}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(epoch + 1, (i+1) * len(input), len(train_data),100. * (i+1) * args.train_batch_size / len(train_data),lossAvg/tnum))
                lossAvg = 0
                tnum=0

        print("Learning rate of epoch %d: %f" % (epoch+1, optimizer.param_groups[0]['lr']))
        print("Now time: "+str(datetime.datetime.now().time()))
        scheduler.step()
        
        if ((epoch + 1) % save_num == 0) or (epoch + 1 == args.epoch_num):
            model_folder = args.model_path + date_str + '/'
            check_makedirs(model_folder)
            torch.save(model.state_dict(), model_folder + 'AESINet-' + str(epoch+1) + '.pth')

    loss_curve(train_counter, train_losses)



def test(args,mID):
    modelID="DPGN-"+mID
    print(modelDATE,modelID,"start!")
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    test_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    to_pil_img = transforms.ToPILImage()

    results_folder = args.results_folder + modelDATE+'/'+modelID+'/'#save path

    test_data = dataset.SemData(split='test', data_root=None, data_list=args.test_list, transform=test_transform)
    img_path_list = test_data.data_list
    img_name_list = []
    n_imgs = len(test_data)
    for i in range(n_imgs):
        img_name = img_path_list[i][1].split('/')[-1]  # img_path_list[i][0] is the image path, img_path_list[i][1] is the gt path
        img_name_list.append(img_name)

    test_sampler = None
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=test_sampler,
                                               drop_last=True)

    #define the network.
    net = basicnet.AESINet(args.block_num, args.block_nod, args.img_dim, args.cov_loop, args.cov_bias, res_pretrained=False)
    device = torch.device(test_device if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    model_dir = args.model_path + modelDATE +'/' + modelID + '.pth'  # the path/file to save the trained model params.
    model.load_state_dict(torch.load(model_dir,map_location='cpu'))
    print('The network parameters are loaded!')

    #test
    model.eval()
    for i, (input, _ , _ , img_size) in enumerate(test_loader):
        input =input.to(device)
        with torch.no_grad():
            sal1, sal2, sal3, sal4, sal5 = model(input)
        n_img, _, _ = sal1.size()
        
        for j in range(n_img):
            salmaps = to_pil_img(sal1[j].cpu())
            salmaps = salmaps.resize((int(img_size[j][1]), int(img_size[j][0])))  # PIL.resize(width, height)
            file_name = img_name_list[i * args.test_batch_size + j]  # get the corresponding image name.
            if not os.path.isdir(results_folder):
                os.makedirs(results_folder)
            salmaps.save(results_folder + file_name)
            if(i * args.test_batch_size + j == 0):
                print("start time:",datetime.datetime.now())
            print('Testing {} th image'.format(i * args.test_batch_size + j))

    print("finish time:",datetime.datetime.now())



if __name__ == '__main__':
    args = get_parser()
    if Flag_train_test == 0:
        criterion = MyLossEdge()
        train(loss_fn=criterion, args=args)
    else:
        for mid in range(test_rangeL,test_rangeR+1):
            test(args=args,mID=str(mid))