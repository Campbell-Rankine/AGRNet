### - Imports - ###
from PIL import Image
import numpy as np
import glob
import cv2 as cv
import os
import random
import argparse
### - other data augmentation imports - ### (if needed)
### - Imports - ###
import math
import numpy as np
import sklearn as sk #general imports, initial data preprocessing/OS stuff
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import os
import torch as T
import torch.nn as nn
import torch.optim as optim #Neural network imports, multiply data etc
from torchvision.transforms import ToTensor
import torchvision.models as models
import torchvision
import torch.nn.functional as F #Neural Network used in Comp4660 at ANU

from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.preprocessing import MinMaxScaler #normalize data
from sklearn.metrics import confusion_matrix #analysis
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

from torchvision.datasets import ImageFolder
from NetworkMain import D, G
from tqdm import tqdm

import tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

"""
TODO: These TODO topics could equally be applied to the FME network
    Remove unnecessary imports
    Tensorboard visualize results (In Bookmark)
    Spacial Transform Networks (Arg defined in parser for some transformation list)
    Optuna experiment analyzing
    Automatic Mixed Precision
"""

#Command Line Functionality
import sys
def process_command_line_arguments() -> argparse.Namespace:
    """Parse the command line arguments and return an object with attributes
    containing the parsed arguments or their default values.
    """
    import json

    parser = argparse.ArgumentParser()

    #Global Params
    parser.add_argument("-l", "--lr", dest="lr", metavar="LR", default = 1e-5,
                        type=str, help="Learning Rate param for both variables")
    parser.add_argument("-g", "--g", dest="grad", metavar="GRAD", default=1,
                        help="Gradient Accumulations technique to save memory. Default: 1")
    parser.add_argument("-ls", "--latentsize", dest="latentsize", metavar="LATENTSIZE", default = 512,
                        type=str, help="Latent size for noise vector to be passed through the Generator during training")
    parser.add_argument("-r", "--res", dest="res", metavar="RES", default=1024,
                        help="Desired output resolution to grow to. Default: 1024")

    #Folder Settings
    parser.add_argument("-d", "--dataset", dest="dataset", metavar="DATASET",
                        type=str, help="Location for all training data for the network (MANDATORY)")
    parser.add_argument("-o", "--output", dest="output", metavar="OUTPUT", default='/output/',
                        help="Output folder (default: %(default)s)")
    parser.add_argument("-resume", "--resume", dest="resume", metavar="RESUME", default=0,
                        help="Resume training and load from a certain set of checkpoints. Format is depth_epoch (Optional)")
    parser.add_argument("-cp", "--cp", dest="cp", metavar="CHECKPOINT", default="/check_points/",
                        help="Checkpoint location folder, to be used when resuming from an epoch")
    parser.add_argument("-w", "--w", dest="weight", metavar="WEIGHT", default="/weight/",
                        help="Weight Location folder. Default: /weight/")

    #TensorBoard Configs
    parser.add_argument("-log", "--log", dest="log", metavar="LOG", default='/Logs/',
                        help="Desired output resolution to grow to. Default: /Logs/")

    #Save args
    #Training resume params
    parser.add_argument("-save", "--save", dest="save", metavar="SAVE", default=-1,
                        help="How many epochs before the network path is saved. Default: -1, meaning it will wait until the end of the epoch to save")
                        

    args = parser.parse_args()
    if not os.path.exists(args.dataset):
        raise SystemExit(
            "Error: Input file '{}' does not exist".format(args.dataset))

    return args

#Load with ImageFolder wrapper
class ImageDataset(Dataset):
    def __init__(self,img_folder,names, transform):
        self.transform = transform
        self.image_names = names #Predetermined dataset
        self.img_folder=img_folder

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        image=Image.open(self.img_folder+self.image_names[index]).convert("RGB")
        image=self.transform(image)
        return image


def Train():
    args = process_command_line_arguments()
    #TensorBoard config
    Log = args.log
    if not os.path.exists(Log):
        os.makedirs(Log)
    #Define output folders
    #root = '/Users/campb/Documents/PersonalProjects/AGRNet/'
    data_dir = args.dataset
    check_point_dir = '/check_points/'
    output_dir = args.output
    weight_dir = args.weight
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    ### - Global Variables - ###
    DFP = args.dataset + '/'
    image_format = 'RGB'
    epochs = 70
    latent_size = args.latentsize
    out_res = args.res
    lr = args.lr
    lambd = 10
    
    #Tensorboard
    writer = SummaryWriter(Log)

    #Start main loop
    device = T.device('cuda:0' if (T.cuda.is_available())  else 'cpu')

    transform = transforms.Compose([
                transforms.Resize(out_res),
                transforms.RandomCrop(out_res, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    print(device)
    #Create networks
    Disc = None
    Gen = None
    Disc = D(latent_size, out_res).to(device)
    Gen = G(latent_size, out_res).to(device)
    
    #Noise for Generator
    fixed_noise = T.randn(16, latent_size, 1, 1).to(device) #Generate grid
    LogFNoise = T.randn(1, latent_size, 1, 1).to(device) #Generate output Generator image
    
    #initialize optimizers
    D_optimizer = optim.Adam(Disc.parameters(), lr=lr, betas=(0, 0.99))
    G_optimizer = optim.Adam(Gen.parameters(), lr=lr, betas=(0, 0.99))
    
    #Metric variables
    D_running_loss = 0.0
    G_running_loss = 0.0
    iter_num = 0

    #assert(os.path.exists(DFP + str(1) + ".jpg"))
    rawimgf = sorted(glob.glob(DFP + '*.jpg', recursive = True))
    ### - image names - ###
    imnames = [i.split('/')[-1].split("t")[-1][0:] for i in rawimgf]

    ### - Global data loader Vars - ###
    norms = (0.5,0.5,0.5), (0.5,0.5,0.5)
    train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((out_res, out_res)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.RandomCrop(out_res, 4)])
    params = {'batch_size': 12,
              'shuffle': True,
             'num_workers': 0,
             'pin_memory': True}
    
    #Evaluation:
    T.backends.cudnn.benchmark = True
    
    #Implement a different train loop that allows us to train the network on some data
    #Depth -> Epochs -> samples for 9 total training cycles
    schedule = [[12, 14, 18, 20, 70, 120, 200, 250, 350], #Epochs for each depth
                [12, 12, 6, 3, 1, 1, 1, 1, 1]] #Batch Size (Has to decrease to conserve memory) 4, 8, 16, 32
    dataset = ImageDataset(DFP, imnames, train_transform)
    GradientAccumulations = int(args.grad)
    data_loader = DataLoader(dataset, **params)
    depth_losses_g = []
    depth_losses_d = []
    
    #DataParallel:
    if T.cuda.device_count() > 1:
        print('Using ', T.cuda.device_count(), 'GPUs')
        D_net = nn.DataParallel(D_net)
        G_net = nn.DataParallel(G_net)
    
    #Resume code
    if not args.resume == 0: #Load params
        print("resuming from " + args.resume)
        Depth, epoch = args.resume.split("_")
        assert(os.path.exists(args.cp+'check_point_depth_%i_epoch_%i.pth' % (int(Depth), int(epoch))))
        print("Path: " + args.cp+'check_point_depth_%i_epoch_%i.pth' % (int(Depth), int(epoch)))
        check_point = T.load(args.cp+'check_point_depth_%i_epoch_%i.pth' % (int(Depth), int(epoch)))
        fixed_noise = check_point['fixed_noise']
        Gen.load_state_dict(check_point['G_net'])
        Disc.load_state_dict(check_point['D_net'])
        G_optimizer.load_state_dict(check_point['G_optimizer'])
        D_optimizer.load_state_dict(check_point['D_optimizer'])
        G_epoch_losses = check_point['G_epoch_losses']
        D_epoch_losses = check_point['D_epoch_losses']
        Gen.depth = check_point['depth']
        Disc.depth = check_point['depth']
        Gen.alpha = check_point['alpha']
        Disc.alpha = check_point['alpha']
        curr_depth = int(args.resume.split("_")[0])
    else:
        curr_depth = 0
    inc = curr_depth
    epoch_ = schedule[0][inc] #Increment 
    batch_s_ = schedule[1][inc]
    tot_iter_num = (len(dataset)/batch_s_)
    Gen.fade_iters = (1-Gen.alpha)/(schedule[0][inc])/(2*tot_iter_num)
    Disc.fade_iters = (1-Disc.alpha)/(schedule[0][inc])/(2*tot_iter_num)
    #mainloop
    for depth in range(curr_depth, int(np.log2(out_res))):
        #Tensorboard
        scaler = T.cuda.amp.GradScaler()
        Gen.train()
        size = 2**(Gen.depth+1)
        print("Training current depth %d at size: %i x %i" % (depth, size, size))
        Depth_loss_g = 0.0
        Depth_loss_d = 0.0
        #Epochs
        epoch_ = schedule[0][inc] #save increment epoch and batch size
        batch_s_ = schedule[1][inc]
        D_epoch_losses = []
        G_epoch_losses = []
        if depth == curr_depth and not curr_depth == 0:
            databar = tqdm(range(int(args.resume.split("_")[1]), epoch_))
        else:
            databar = tqdm(range(epoch_))
        for epoch in databar:
            T.cuda.empty_cache()
            D_optimizer.zero_grad() #Memory handling
            G_optimizer.zero_grad()
            D_epoch_loss = 0.0
            G_epoch_loss = 0.0

            #databar = tqdm(data_loader)
            for i, samples in enumerate(data_loader):
                ##  update D
                if size != out_res: #Basically need to, A Reshape, B prepare the data for the networks
                    with T.no_grad():
                        samples = F.interpolate(samples, size=(size,size))
                samples = samples.to(device)

                noise = T.randn(samples.size(0), latent_size, 1, 1, device=device)
                with T.cuda.amp.autocast():
                    fake = Gen(noise)
                    #out_grid = make_grid(fake, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**Gen.depth))).permute(1,2,0)
                    #plt.imshow(out_grid.cpu())
                    fake_out = Disc(fake.detach())
                    real_out = Disc(samples) #Should add noise to these images
                ## Gradient Penalty

                eps = T.rand(samples.size(0), 1, 1, 1, device=device)
                x_hat = eps * samples + (1 - eps) * fake.detach()
                x_hat.requires_grad = True
                with T.no_grad():
                    eps = eps.expand_as(Gen(noise))
                with T.cuda.amp.autocast():
                    px_hat = Disc(x_hat)
                scaled_grad = T.autograd.grad(
                                            outputs = scaler.scale(px_hat.sum()),
                                            inputs = x_hat, 
                                            create_graph=False
                                            )[0]

                invscale = 1./scaler.get_scale()
                grad = scaled_grad * invscale
                with T.cuda.amp.autocast():
                    grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
                    gradient_penalty = lambd * ((grad_norm  - 1)**2).mean()
                    grad = None
                    D_loss = (fake_out.mean() - real_out.mean() + gradient_penalty) / GradientAccumulations
                ###########
                #Apply gradient clipping to both 


                now = datetime.now()
                epochwriter = SummaryWriter(Log + '_epoch_' + str(epoch))
                epochwriter.add_scalar("loss/discriminator", D_loss, i)
                scaler.scale(D_loss).backward()
                scaler.unscale_(D_optimizer)
                nn.utils.clip_grad_value_(Disc.parameters(), clip_value=1.0)
                if (i+1) % GradientAccumulations == 0:
                    scaler.step(D_optimizer)
                    scaler.update()
                    Disc.zero_grad()


                ##	update G
                fake_out = Disc(fake)
                with T.cuda.amp.autocast():
                    G_loss = (- fake_out.mean())  / GradientAccumulations
                epochwriter.add_scalar("loss/generator", G_loss, i)
                epochwriter.flush()
                scaler.scale(G_loss).backward()
                scaler.unscale_(G_optimizer)
                nn.utils.clip_grad_value_(Gen.parameters(), clip_value=1.0)
                if (i+1) % GradientAccumulations == 0:
                    scaler.step(G_optimizer)
                    scaler.update()
                    Gen.zero_grad()

                ##############
                D_running_loss += abs(D_loss.item())
                G_running_loss += abs(G_loss.item())
                iter_num += 1
                if i % 3== 0:
                    D_running_loss /= iter_num
                    G_running_loss /= iter_num
                    #print('iteration : %d, gp: %.2f' % (i, gradient_penalty))
                    databar.set_description('Size: %0.3f Epoch: %.3f D_loss: %.3f   G_loss: %.3f' % (size, epoch, D_running_loss ,G_running_loss))
                    iter_num = 0
                    D_running_loss = 0.0
                    G_running_loss = 0.0

            #plot gradients

            D_epoch_losses.append(D_epoch_loss/tot_iter_num)
            G_epoch_losses.append(G_epoch_loss/tot_iter_num)


            check_point = {'G_net' : Gen.state_dict(), 
                           'G_optimizer' : G_optimizer.state_dict(),
                           'D_net' : Disc.state_dict(),
                           'D_optimizer' : D_optimizer.state_dict(),
                           'D_epoch_losses' : D_epoch_losses,
                           'G_epoch_losses' : G_epoch_losses,
                           'fixed_noise': fixed_noise,
                           'depth': Gen.depth,
                           'alpha':Gen.alpha
                           }
            with T.no_grad():
                Gen.eval()
                out_imgs = Gen(fixed_noise).to(device)
                out_grid = make_grid(out_imgs, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**Gen.depth))).permute(1,2,0)
                plt.imshow(out_grid.cpu())
                plt.savefig(output_dir + 'size_%i_epoch_%d' %(size ,epoch))
                if int(args.save) != -1:
                    if epoch % int(args.save) == 0:
                        with T.no_grad():
                            T.save(check_point, args.cp + 'check_point_depth_%d_epoch_%d.pth' % (depth, epoch))
                            T.save(Gen.state_dict(), weight_dir + 'G_weight_depth_%d_epoch_%d.pth' % (depth, epoch))
        depth_losses_g.append(np.mean(G_epoch_losses))
        depth_losses_d.append(np.mean(D_epoch_losses))
        #Increment depth step
        if int(args.save) == -1:
            with T.no_grad():
                T.save(check_point, args.cp + 'check_point_depth_%d_epoch_%d.pth' % (depth, epoch))
                T.save(Gen.state_dict(), weight_dir + 'G_weight_depth_%d_epoch_%d.pth' % (depth, epoch))
        if 2**(Gen.depth+2) <= out_res:
            #Log for Tensorboard
            now = datetime.now()
            depthwriter = SummaryWriter(Log + '_depth_')
            out_img = Gen(LogFNoise).to(device)
            grid = torchvision.utils.make_grid(out_img, normalize=True)
            depthwriter.add_image("out/generator", grid, depth)
            depthwriter.flush()

            #Upgrade net
            inc += 1
            print("Growing network to size: " + str(2**(Gen.depth+2)))
            data_loader = DataLoader(dataset, **params)
            tot_iter_num = tot_iter_num = (len(dataset)/batch_s_)
            Gen.inc_depth(schedule[0][inc]*tot_iter_num)
            Disc.inc_depth(schedule[0][inc]*tot_iter_num)
            size = 2**(Gen.depth+1)
            print("Output Resolution: %d x %d" % (size, size))

def main():
    Train()
            
if __name__ == '__main__':
    main()
