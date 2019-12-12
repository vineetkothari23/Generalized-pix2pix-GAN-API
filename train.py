import torch 
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import os
from PIL import Image
import glob
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import random
from bokeh.io import curdoc, show, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen
import time
import pickle
from tqdm import tqdm
from utils import show_result,show_train_hist,generate_animation,data_load,imgs_resize, random_crop, random_fliplr
from data_loader import localImageDataset
from model import generator, discriminator, normal_init

def train_process(config):
  
  root_dir=config.root_dir
  inp_width, inp_height, inp_channels,train_split=config.inp_width, config.inp_height, config.inp_channels,config.train_split
  # model parameters
  lrG=config.lrG
  lrD=config.lrD
  beta1=config.beta1
  beta2=config.beta2
  L1_lambda=config.L1_lambda
  ngf=config.ngf
  ndf=config.ndf

  dataset=localImageDataset(root_dir, inp_width, inp_height, inp_channels)
  print("Length of dataset: ",len(dataset))
  train_size=int(train_split*len(dataset))
  val_size=len(dataset)-train_size
  train_dataset, val_dataset=torch.utils.data.random_split(dataset,[train_size,val_size])
  train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size,
                                              shuffle=True,
                                            num_workers=4)
  num_batches=len(train_dataloader)
  val_dataloader=torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                            num_workers=4)
  
  #from model import generator, discriminator
  #import utils



  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  G = generator(ngf)
  D = discriminator(ndf)
  BCE_loss=nn.BCELoss().cuda()
  L1_loss=nn.L1Loss().cuda()
  G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
  D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))
  start_time=time.time()
  epoch_start=0
  epoch_end=epoch_start+train_epoch

  #loss


  if(os.path.isfile(model_dir+'generator_param.pkl') and os.path.isfile(model_dir+'discriminator_param.pkl')):
    
    G_checkpoint=torch.load(model_dir+'generator_param.pkl',map_location=device)
    D_checkpoint=torch.load(model_dir+'discriminator_param.pkl',map_location=device)
    G.load_state_dict(G_checkpoint['model_state_dict'])
    D.load_state_dict(D_checkpoint['model_state_dict'])
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    G_optimizer.load_state_dict(G_checkpoint['optimizer_state_dict'])
    D_optimizer.load_state_dict(D_checkpoint['optimizer_state_dict'])
    
    train_hist=G_checkpoint['train_hist']
    epoch_start=G_checkpoint['epoch']
    epoch_end=epoch_start+train_epoch
  else:
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    
    G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
    D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))

    train_hist={}
    train_hist['D_losses']=[]
    train_hist['G_losses']=[]
    train_hist['per_epoch_ptimes']=[]
    train_hist['total_ptime']=[]
    epoch_end=epoch_start+train_epoch



  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  G = generator(ngf)
  D = discriminator(ndf)
  G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
  D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))
  #loss
  BCE_loss=nn.BCELoss().to(device)
  L1_loss=nn.L1Loss().to(device)

  if(os.path.isfile(model_dir+'generator_param.pkl') and os.path.isfile(model_dir+'discriminator_param.pkl')):
    
    G_checkpoint=torch.load(model_dir+'generator_param.pkl',map_location=device)
    D_checkpoint=torch.load(model_dir+'discriminator_param.pkl',map_location=device)
    G.load_state_dict(G_checkpoint['model_state_dict'])
    D.load_state_dict(D_checkpoint['model_state_dict'])
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    #D.eval()

    G_optimizer.load_state_dict(G_checkpoint['optimizer_state_dict'])
    D_optimizer.load_state_dict(D_checkpoint['optimizer_state_dict'])
    
    train_hist=G_checkpoint['train_hist']
    epoch_start=G_checkpoint['epoch']
    epoch_end=epoch_start+train_epoch
  else:
    print("Previous model not found. Restarting train process...")
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    
    
    G_optimizer=optim.Adam(G.parameters(),lr=lrG,betas=(beta1,beta2))
    D_optimizer=optim.Adam(D.parameters(),lr=lrD,betas=(beta1,beta2))

    train_hist={}
    train_hist['D_losses']=[]
    train_hist['G_losses']=[]
    train_hist['per_epoch_ptimes']=[]
    train_hist['total_ptime']=[]
    epoch_start=0
    epoch_end=epoch_start+train_epoch


  for epoch in range(epoch_start,epoch_end):
    D_losses=[]
    G_losses=[]
    epoch_start_time=time.time()
    num_iter=0
    for text_image, inp_image in train_dataloader:
      inp_image,text_image=Variable(inp_image.to(device)),Variable(text_image.to(device))
      D.zero_grad()

      
      D_result=D(inp_image,text_image).squeeze()
      D_real_loss=BCE_loss(D_result,Variable(torch.ones(D_result.size()).to(device)))
      
      G_result=G(inp_image)
      D_result=D(inp_image,G_result).squeeze()
      D_fake_loss=BCE_loss(D_result,Variable(torch.zeros(D_result.size()).to(device)))
      
      D_train_loss=(D_real_loss +D_fake_loss)*0.5
      D_train_loss.backward()
      D_optimizer.step()
      train_hist['D_losses'].append(float(D_train_loss))
      
      D_losses.append(float(D_train_loss))
      D_losses.append(float(0))
      
      #training generator
      G.zero_grad()

      G_result=G(inp_image)
      D_result=D(text_image,G_result).squeeze()

      G_train_loss=BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device))) + L1_lambda*L1_loss(G_result,text_image)
      G_train_loss.backward()
      G_optimizer.step()

      train_hist['G_losses'].append(float(G_train_loss))
      G_losses.append(float(G_train_loss))
      num_iter+=1

    torch.save({
              'epoch': epoch,
              'model_state_dict': G.state_dict(),
              'optimizer_state_dict': G_optimizer.state_dict(),
              'train_hist': train_hist
              }, model_dir+'generator_param.pkl')

    torch.save({
              'model_state_dict': D.state_dict(),
              'optimizer_state_dict': D_optimizer.state_dict(),
              },model_dir+'discriminator_param.pkl')

    epoch_end_time=time.time()
    per_epoch_ptime=epoch_end_time-epoch_start_time
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                torch.mean(torch.FloatTensor(G_losses))))
    fixed_p =  output_dir  + str(epoch + 1) + '.png'
    #show_result(G, Variable(inp_image.to(device), volatile=True), text_image.cpu(), (epoch+1), save=True, path=fixed_p)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
  end_time=time.time()
  total_ptime=end_time-start_time
  train_hist['total_ptime'].append(total_ptime)
  print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    

  with open(report_dir+'train_hist.pkl', 'wb') as f:
      pickle.dump(train_hist, f)

  show_train_hist(train_hist, save=True, path=report_dir + 'train_hist.png')

  