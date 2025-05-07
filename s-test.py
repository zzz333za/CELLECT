


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:25:14 2022

@author: zzz333-pc
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import glob

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import Dataset
import random
from tqdm import tqdm as ttm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import warnings
from random import randint
warnings.filterwarnings("ignore")
import numpy as np
from skimage import io

import pandas as pd
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import label,regionprops
import cv2
import numpy as np
from PIL import Image
from tqdm  import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage import data, util
from skimage.measure import label,regionprops
from recoloss import CrossEntropyLabelSmooth,TripletLoss

import random
from unetext3Dn_con7s import UNet3D

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Training script for the model")

# 添加参数
parser.add_argument('--data_dir', type=str, required=True, help="Path to the training data directory")
parser.add_argument('--out_dir', type=str, required=True, help="Path to the output data directory")
parser.add_argument('--cpu', action='store_true', help="Use CPU instead of GPU if this flag is set.")

parser.add_argument('--model1_dir', type=str, required=True, help="Path to the Unet model")
parser.add_argument('--model2_dir', type=str, required=True, help="Path to MLP model1")
parser.add_argument('--model3_dir', type=str, required=True, help="Path to MLP model2")




# 解析参数
args = parser.parse_args()
datap=args.data_dir
outp=args.out_dir
mp1=args.model1_dir
mp2=args.model2_dir
mp3=args.model3_dir



def fill(n,x,y,z,v=1,s=4):
    rr,cc=draw.ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
    z1=int(max(0,z-3))
    z2=int(min(100,z+4))

    if z1==z2:
        n[rr,cc,z1]=v
    else:
        n[rr,cc,z1:z2]=v
    return n
def ar(v):
    av=v.clone()
    av=torch.zeros([26,v.shape[0],v.shape[1],v.shape[2]])
    zv=torch.zeros([v.shape[0]+2,v.shape[1]+2,v.shape[2]+2])
    zv[1:-1,1:-1,1:-1]=v
    n=0
    
    for x in range(3):
        for y in  range(3):
            for z in range(3):
                if not x==1 and y==1 and z==1:
                    av[n]=zv[x:x+v.shape[0],y:y+v.shape[1],z:z+v.shape[2]]
                    n=n+1
    return av
def tim(a):
    l=[]
    a.seek(0)
    l.append(np.array(a)[:,:,np.newaxis])
    n=0
    while(1):
       n+=1
       try:
           a.seek(n)
       except:break
       l.append(np.array(a)[:,:,np.newaxis])
    b=np.concatenate(l,2)
    return b
############## Prepare the target dataset
win1=[128,150,6]   

op='2'
vl=os.listdir(datap+'/mskcc_confocal_s'+op+'/images/')
vl=[i for i in vl if 'tif' in i]

vD={}
for i in tqdm(vl):
    if 'tif' in i:         
        
        num=int(i.split('_')[-1].split('.')[0][1:])
        #a=Image.open('D:/mskcc-confocal/mskcc_confocal_s1/images/'+i)
        if num<273 :
            
            vD[num]=datap+'/mskcc_confocal_s'+op+'/images/'+i
        

vd1=pd.read_table(datap+'/mskcc_confocal_s'+op+'/tracks/tracks.txt')#,skiprows=3)
vd2=pd.read_table(datap+'/mskcc_confocal_s'+op+'/tracks/tracks_polar_bodies.txt')#,skiprows=3)

################### Configure the data loader
class VDataset(Dataset):

    def __init__(self, data,le, transform=None):
        
        #self.path = path
        self.data =data
        self.transform = transform
       
    def __len__(self):
        
        return len(self.data)-5#-250#-220#-150#-220#200-3

    def __getitem__(self, i):
        
   
        
        j=i#+250#+150#+220#+200
        g=vD[j]

        o=1
        g2=vD[j+o]
  
        g3=vD[j+o+1]
        #j=random.choice(x_train)
        op=str(2)
      
 
        img=tim(Image.open(g))
        img2=tim(Image.open(g2))
        img3=tim(Image.open(g3))
        #img4=tim(Image.open(vD[j+3]))
        #t=np.load('D:/mskcc-confocal/ls'+op+'/'+str(j)+'-k1-3d-1-imaris.npy')
       
      
        b = torch.from_numpy((img.astype(float)))
        
  
       
        img3 = torch.from_numpy(img3.astype(float))
      
    
        c=torch.from_numpy(img2.astype(float))

        #t9[t9>0]+=100000
        
     
        #size1=torch.from_numpy(t31)
        #size2=torch.from_numpy(t32)
        #print(b.shape,timg.shape,k)
        return {'image': b, 'im2':c,'img3':img3} 



############################### Define the MLP model

class EXNet(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(EXNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        

        self.l1=nn.Linear(self.in_channels*5, self.in_channels)
        self.l2=nn.Linear(self.in_channels*5, self.in_channels)
        self.l3=nn.Linear(self.in_channels*5, self.in_channels)
        self.l4=nn.Linear(3*5, self.in_channels)
        self.l5=nn.Linear(6*5, self.in_channels)
        self.l6=nn.Linear(6*5, self.in_channels)
        self.bnx=nn.BatchNorm1d(self.in_channels)
        self.bny=nn.BatchNorm1d(self.in_channels)
        self.bn1=nn.BatchNorm1d(self.in_channels)
        self.bn2=nn.BatchNorm1d(self.in_channels)
        self.bn3=nn.BatchNorm1d(self.in_channels)
        self.bn4=nn.BatchNorm1d(self.in_channels)
        self.bn5=nn.BatchNorm1d(self.in_channels)
        self.bn6=nn.BatchNorm1d(self.in_channels)
        self.bn7=nn.BatchNorm1d(self.in_channels*2)
        self.bnc=nn.BatchNorm2d(8)
        self.fc=nn.Linear(self.in_channels*6, self.in_channels*2)
        self.fc2=nn.Linear(self.in_channels*2, self.in_channels)
        self.out=nn.Linear(self.in_channels, self.n_classes)
        self.c1 = nn.Conv2d(1, 8, kernel_size=(2,5), stride=1, padding=(0,2),
                                     bias=False)
        self.c2 = nn.Conv2d(1, 8, kernel_size=(2,3), stride=1, padding=(0,1),
                                     bias=False)
        self.cx=nn.Linear(self.in_channels*8*2, self.in_channels)
        '''self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)'''
        self.outc=nn.Linear(self.in_channels+self.n_classes, 2)
        self.d1= nn.Dropout(p=0.1)
        self.d2= nn.Dropout(p=0.1)
        self.s=nn.Sigmoid()
    def forward(self, x,y,p,s,zs):
        #  Level 1 context pathway
      
        x01=torch.cat([x,y[:,0].unsqueeze(1)],1)

        x01=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x01.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x01.unsqueeze(1))))],1).reshape(x01.shape[0],-1))
  
        x02=torch.cat([x,y[:,1].unsqueeze(1)],1)

        x02=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x02.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x02.unsqueeze(1))))],1).reshape(x02.shape[0],-1))
        x03=torch.cat([x,y[:,2].unsqueeze(1)],1)

        x03=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x03.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x03.unsqueeze(1))))],1).reshape(x03.shape[0],-1))
        x04=torch.cat([x,y[:,3].unsqueeze(1)],1)

        x04=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x04.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x04.unsqueeze(1))))],1).reshape(x04.shape[0],-1))
        x05=torch.cat([x,y[:,4].unsqueeze(1)],1)

        x05=self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x05.unsqueeze(1)))),self.lrelu(self.bnc(self.c2(x05.unsqueeze(1))))],1).reshape(x05.shape[0],-1))
   
        x1=torch.cat([x01,x02,x03,x04,x05],1).reshape(x.shape[0],-1)
        x2=((y-x)).reshape(x.shape[0],-1)
        #x3=(self.bnx(x)*self.bny(y)).reshape(x.shape[0],-1)
        xx=torch.sqrt((x*x).sum(2))
        yy=torch.sqrt((y*y).sum(2))
        xy=x*y
        xy2=(xx*yy).unsqueeze(-1)#.repeat(1,1,xy.shape[-1])
        #print(xy.shape,xy2.shape)
        x3=xy/xy2
        x3=x3.reshape(x.shape[0],-1)
        
        x4=p.reshape(x.shape[0],-1)
        x5=s.reshape(x.shape[0],-1)
        x6=zs.reshape(x.shape[0],-1)
        x1=self.lrelu(self.bn1(self.l1(x1)))
        x2=self.lrelu(self.bn2((self.l2(x2))))
        x3=self.lrelu(self.bn3(self.l3(x3)))
        x4=self.lrelu(self.bn4((self.l4(x4))))
        x5=self.lrelu(self.bn5((self.l5(x5))))
        x6=self.lrelu(self.bn6((self.l6(x6))))
        nx=torch.cat([x1,x2,x3,x4,x5,x6],1)
        #x1=self.lrelu(self.l1(x))
        ui=self.bn7(self.lrelu(self.fc(((nx))) ))
        nx=self.lrelu(self.fc2(ui))
        f=(self.out(nx))   
        #fk=self.outc(torch.cat([nx,f],1))   
        return f#,fk

########################################## Load the model

SMOOTH = 1e-6

n_epochs = 150
batch_size =1
os.environ['CUDA_VISIBLE_DEVICES']='0'
if args.cpu:
    device = 'cpu'
else:
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

#device = torch.device("cuda:0")

from skimage import draw

tm=2
tm1=1

#G =UNet(1,1)
U =UNet3D(2,6)
EX=EXNet(64,8)
EN=EXNet(64,6)

#U2 =UNet(1,2)
#model.fc = torch.nn.Linear(2048, 6)

#G.to(device)
U.to(device)
EX.to(device)
EN.to(device)

if device=='cpu':
    U.load_state_dict(torch.load(mp1), map_location='cpu')
    EX.load_state_dict(torch.load(mp2), map_location='cpu')
    EN.load_state_dict(torch.load(mp3), map_location='cpu')
else:
    U.load_state_dict(torch.load(mp1))
    EX.load_state_dict(torch.load(mp2))
    EN.load_state_dict(torch.load(mp3))


print('    Total params: %.2fM' % (sum(p.numel() for p in U.parameters())/1000000.0))


l0=[]

l1=[]

#model=torch.load('../model/2db.pth')

val_dataset = VDataset(
 vD, le=1000)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0) 
val_loss=1000
va=10000
win=3
lg=32
ko=3
pl=16
pk=(1,1,1)


############################################### Define search masks

def kfbx(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],45,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               if np.abs(xc)+max(np.abs(y),np.abs(z))<=2:
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
                    #print(num)
    return kq
def kfbu(x):   
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],25,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               if np.abs(xc)+np.abs(y)+np.abs(z)<=2:
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
                    
    return kq
def kf(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
                    num+=1
    return kq
def kfb(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]+4])
    kq=torch.zeros([x.shape[0],125,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,2:-2]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
            for z in [-2,-1,0,1,2]:
               
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],z+2:z+2+x.shape[3]]
                    num+=1
    return kq
def kfl(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                if np.abs(xc)+np.abs(y)+np.abs(z)<=1:
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
                    num+=1
    return kq
def kflb(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]+2])
    kq=torch.zeros([x.shape[0],27,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,1:-1]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                if np.abs(xc)+max(np.abs(y),np.abs(z))<=1:
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],z+1:z+1+x.shape[3]]
                    num+=1
    return kq
def kfs2(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+4,x.shape[2]+4,x.shape[3]])
    kq=torch.zeros([x.shape[0],25,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,2:-2,2:-2,:]+=x.cpu()
    num=0
    for xc in [-2,-1,0,1,2]:
        for y in [-2,-1,0,1,2]:
               
                    kq[:,num]=kn[:,xc+2:xc+2+x.shape[1],y+2:y+2+x.shape[2],:]
                    num+=1
    return kq
def kfs(x):
    
    kn=torch.zeros([x.shape[0],x.shape[1]+2,x.shape[2]+2,x.shape[3]])
    kq=torch.zeros([x.shape[0],9,x.shape[1],x.shape[2],x.shape[3]])
    kn[:,1:-1,1:-1,:]+=x.cpu()
    num=0
    for xc in [-1,0,1]:
        for y in [-1,0,1]:
        
                    kq[:,num]=kn[:,xc+1:xc+1+x.shape[1],y+1:y+1+x.shape[2],:]
                    num+=1
    return kq



# Data preprocessing
def PA(x):
    #ma=x.max()
    ##mi=x.min()
    #x1=(mi+100)
    #x[x<x1]=x1
    #x=x-x.min()
    #x=x+1900
    #x[x<2000]=2000
    #x=x-100
    x=torch.log1p(x)
    return x

tk0 = tqdm(data_loader_val, desc="Iteration")
   
U.eval()
EN.eval()
EX.eval()


DK=np.zeros([tm*2+1,tm*2+1,9])
rr,cc=draw.ellipse(tm,tm, tm,tm)
DK[rr,cc,:]=1
tm=2
tm1=1
DK=torch.from_numpy(DK).float().to(device).reshape([1,1,tm*2+1,tm*2+1,9])
DK1=np.array([[0,0,1,1,1,0,0],[0,1,1,2,1,1,0],[1,1,2,3,2,1,1],[1,2,3,4,3,2,1],[1,1,2,3,2,1,1],[0,1,1,2,1,1,0],[0,0,1,1,1,0,0]])
DK1=torch.from_numpy(DK1).float().to(device).reshape([1,1,7,7,1])#.repeat(1,1,1,1,3)
def ud(x):
    x1=torch.zeros_like(x)
    x2=torch.zeros_like(x)
    x1[:,:,:,:,:-1]=x[:,:,:,:,1:]
    x2[:,:,:,:,1:]=x[:,:,:,:,:-1]
    x0=x==0
    xn=(x0*torch.cat([x1,x2],1).max(1)[0]-1)
    
    xn[xn<0]=0
    return x+xn

######################## Test each patch using U-Net
def vsup256(U,x,la):
   
    Uout,uo,foz,zs1,size1 =  U(x)
    Uf=foz.transpose(0,1)
    
    uc=((kflb(uo[:,:].max(1)[0]).to(device).max(1)[0]==uo[:,4]))*(Uout.argmax(1)==1)#*(size1[:,0]>1)
    #uc=((kflb(uo[:,:].max(1)[0]).to(device).max(1)[0]==uo[:,4]))*(kfl(Uout.argmax(1)==0).to(device).sum(1)==0)
    uc[:,:3]=0
    uc[:,252:]=0
    uc[:,:,:3]=0
    uc[:,:,252:]=0
    uc[:,:,:,:3]=0
    uc[:,:,:,30:]=0    
    ar=F.conv3d((uc>0).float().unsqueeze(1), DK1, padding=(3,3,0))
    
    ar=F.conv3d((ar>0).float(), DK1, padding=(3,3,0))
    for i in range(1):
            ar=ud(ar)
             
    u,x,y,z=torch.where(uc.to(device, dtype=torch.float))

    #print(u.shape,lx,lx1)
    f1=Uf[:,u,x,y,z].transpose(0,1)
    s1=size1[u,0,x,y,z]
    zs1=F.sigmoid(zs1[u,:,x,y,z])
    #s11=size1[u,0,x,y,z+1]
    #s12=size1[u,0,x,y,z-1]
    #s1=torch.cat([s1.unsqueeze(1),s11.unsqueeze(1),s12.unsqueeze(1)],1).max(1)[0]
    
    lb1=la[u,x,y,z]
   
    p1=torch.cat([u.unsqueeze(1),x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],1)
    #p2=torch.cat([u2.unsqueeze(1),x2.unsqueeze(1),y2.unsqueeze(1),z2.unsqueeze(1)],1)
  
    #Uout=torch.cat([Uout[:,:,:,:240],Uoutf[:,:,:,240-(320-256):]],3)
    return p1,f1,s1,zs1,lb1,ar,Uout
def cp2(x,ba,bb,bc):
    aa=((x[:,1])<=ba.max())*((x[:,1])>=ba.min())
    ab=((x[:,2])<=bb.max())*((x[:,2])>=bb.min())
    ac=((x[:,3])<=bc.max())*((x[:,3])>=bc.min())
    return aa*ab*ac
def cp3(x,ba):
    u,x,y,z=x[:,0],x[:,1],x[:,2],x[:,3]
    na=ba[x,y,z]>0
    return na

########## U-Net feature extraction process for the entire image
def bvsup256(U,x,l):
    u,c,pl1,pl2,pl3=x.shape
    pol=256
    polz=32
    pl=pol-8#256-128
    plz=polz-4
    xm,ym,zm=pl1//pl+int(pl1%pl>0),pl2//pl+int(pl2%pl>0),pl3//plz+int(pl3%plz>0)
    p,f,s,zs,lb=[],[],[],[],[]
    plist=[]
    num=0
    uout=torch.zeros([pl1,pl2,pl3])
    ku=torch.zeros([pl1,pl2,pl3])
    kup=torch.zeros([pl1,pl2,pl3])
    #print(x[:,:,:,:256,:].shape,x[:,:,:,:256,:].max(),x[:,:,:,:256,:].min(),x[:,:,:,-256:,:].shape)
    for x1 in range(xm):
        for y1 in range(ym):
            for z1 in range(zm):
                num+=1
                ku[ku>1]=1
                
                v1,v2,v3,v4,v5,v6=x1*pl,x1*pl+pol,y1*pl,y1*pl+pol,z1*plz,z1*plz+polz
                if v2>=pl1:
                    v2=pl1
                    v1=v2-pol
                if v4>=pl2:
                    v4=pl2
                    v3=v4-pol
                if v6>=pl3:
                    v6=pl3
                    v5=v6-polz                               
                #print(v1,v2,v3,v4,v5,v6)
                with torch.no_grad():
                    p1,f1,s1,zs1,lb1,ar,uo=  vsup256(U,x[:,:,v1:v2,v3:v4,v5:v6],l[:,v1:v2,v3:v4,v5:v6])  
                uout[v1+5:v2-5,v3+5:v4-5,v5:v6]+=(uo.argmax(1).squeeze().cpu()[5:-5,5:-5]==1).float()
                    
                ar[ar>1]=1
                ar=ar.squeeze().cpu()
                #kup[v1:v2,v3:v4,v5:v6]+=ar
                #ku[v1:v2,v3:v4,v5:v6]+=ar
                #kux=(ku>1).float()*kup
                p1[:,1]+=v1
                p1[:,2]+=v3
                p1[:,3]+=v5
                if num>1:
                    
              
                  
                   
                    kn=cp3(p1.cpu(),ku==1)
                    
           
            
                    fn=~kn
                    p1=p1[fn]
                    f1=f1[fn]
                    s1=s1[fn]
                    zs1=zs1[fn]
                    lb1=lb1[fn]
           
                    p=torch.cat([p,p1],0)
                    f=torch.cat([f,f1],0)
                    s=torch.cat([s,s1],0)
                    zs=torch.cat([zs,zs1],0)
                    lb=torch.cat([lb,lb1],0)
                else:
                    p=p1
                    f=f1
                    s=s1
                    zs=zs1
                    lb=lb1    
                ku[v1:v2,v3:v4,v5:v6]+=ar
    #print(p.shape[0],torch.unique(l).shape[0],torch.unique(lb).shape[0])
    #Uout=torch.cat([Uout[:,:,:,:240],Uoutf[:,:,:,240-(320-256):]],3)
    return p,f,zs,s,lb,uout
################################# Load and test the final frame to determine the processing scope

img3=tim(Image.open(vD[269])).astype(float)
img4=tim(Image.open(vD[270])).astype(float)
labels=torch.zeros(img3.shape).unsqueeze(0).to(device)
with torch.no_grad():
    p1,f1,zs1,s1,lb1,uo =  bvsup256(U,(PA(torch.cat([torch.from_numpy(img3).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0),torch.from_numpy(img4).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0)],1))),labels)
def box(x):
   
    x1=(x>1)
    l1=list(x1.sum((0,1))>10)
    z1=l1.index(1)
    z2=len(l1)-l1[::-1].index(1)
    l1=list(x1.sum((1,2))>1)
    xx1=l1.index(1)
    xx2=len(l1)-l1[::-1].index(1)
    l1=list(x1.sum((0,2))>1)
    y1=l1.index(1)
    y2=len(l1)-l1[::-1].index(1)

    return xx1,xx2,y1,y2,z1,z2#, mi+(ma-mi)/10
x1,x2,y1,y2,z1,z2=box(uo.cpu())
aq=img3[uo>=1].min()

win2=[max(2,x1-10),max(2,y1-10),2]
pla=max(256,x2-x1)+10
plb=max(256,y2-y1)+10
pl=38#max(38,z2-z1)
################################## Update preprocessing

def PA(x):
    #ma=x.max()
    ##mi=x.min()
    #x1=(mi+100)
    x[x<aq]=aq
    x=x-x.min()
    x=x+1900
    #x[x<2000]=2000
    #x=x-100
    x=torch.log1p(x)
    return x
###############################Testing
edge=[]
FP=[]
FN=[]
SIS=[]
FPD=[]
FND=[]
val_loss=0
kk=0
se=0
iou=0
n=0
DP={}
TP={}
De=0
DD={}
l=[]
l2=[]
ll=[]
k=[]
ni=0
cuo={}
ER=[]
ERT={}
jER={}
nid=0
ZZ={}
PZ={}
for step, batch in enumerate(tk0):
    #step+=150
    #Data loading
    
    inputs = batch["image"].unsqueeze(1)
   
    in2=batch["im2"].unsqueeze(1)
   
    inputs = inputs[:,:,win2[0]:win2[0]+pla,win2[1]:win2[1]+plb,win2[2]:win2[2]+pl].to(device, dtype=torch.float)
   
    in2=in2[:,:,win2[0]:win2[0]+pla,win2[1]:win2[1]+plb,win2[2]:win2[2]+pl].to(device, dtype=torch.float)
    
    in3= batch["img3"].unsqueeze(1)[:,:,win2[0]:win2[0]+pla,win2[1]:win2[1]+plb,win2[2]:win2[2]+pl].to(device, dtype=torch.float)
    #U-net process
    with torch.no_grad():
                if step==0:
                    #Uout,uo,fo,zs1,size1 =  U(PA(torch.cat([inputs,in2],1)))
                    p1,f1,zs1,s1,lb1,_ =  bvsup256(U,(PA(torch.cat([inputs,in2],1))),inputs[:,0]*0)
                  
                    
                else:
                    #print(kel)
                    #dfghjk
                    p1=zp2#[kel]
                    f1=zf2#[kel]
                    zs1=zzsk#[kel]
                    s1=zzs2#[kel]
                    lb1=labels[p1[:,0],p1[:,1],p1[:,2],p1[:,3]]
                    #weqwe
                #Uout2,uo2,fo2,zs2,size2 =  U(PA(torch.cat([inputs,in2],1)))
                p2,f2,zs2,s2,lb2,_ =  bvsup256(U,(PA(torch.cat([in2,in3],1))),in2[:,0]*0) 
                '''uu=s2>1
                p2=p2[uu]
                f2=f2[uu]
                zs2=zs2[uu]
                s2=s2[uu]
                lb2=lb2[uu]'''
    zp2=p2.clone()
    zf2=f2.clone()
    zzsk=zs2.clone()
    zzs2=s2.clone()
  

 
    p1=p1[:,1:]
    p2=p2[:,1:]
    p1[:,2]=p1[:,2]*5
    p2[:,2]=p2[:,2]*5           
 
    #lb2=lb2[pq]
    #s2=s2[pq]     
    qf,ql,px,gf,gl,py=f1,lb1,p1,f2,lb2,p2

    if  qf.shape[0]>2 and gf.shape[0]>2:

##############################################################
        # Use the MLP to find points belonging to the same cell within the same frame and exclude them

        if step>0:
            dx=dx2m
            fq=eq.clone()
        else:

            m, n = px.shape[0], px.shape[0]
            px1=px.clone()
            px1[:,2]=px1[:,2]//5
            distmat = torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, px1.float(),px1.float().t())
            qx,q=distmat.topk(min(6,px1.shape[0]),largest=False)
            
            if q.shape[1]<6:
                while q.shape[1]<6:
                    q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                    qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)   
            qx=qx[:,1:]
            q=q[:,1:]
                           
       
            ey=[]
            ep=[]
            es=[]
            ezs=[]
            #epx=px*px
       
            for jk in range(5):
                ey.append(qf[q[:,jk].unsqueeze(1)])
                t=px[q[:,jk]]-px
                ep.append(torch.abs(t).unsqueeze(1))
            epy=torch.cat(ep,1).float()
            ey=torch.cat(ey,1)
            epyy=torch.sqrt((epy*epy).sum(-1))
        
            for jk in range(5):
                ts1=s1[q[:,jk]]
                es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                     epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
                tzs1=zs1[q[:,jk]]
                ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
       
            esp=torch.cat(es,1)
            ezsp=torch.cat(ezs,1)
            with torch.no_grad():
                score=F.sigmoid(EN(qf.unsqueeze(1),ey,epy.float(),esp,ezsp))
            yl=[]
         
            for jk in range(5):
                yl.append((ql[q[:,jk]]).unsqueeze(1))
            yl=torch.cat(yl,1) 
            ss=(score[:,:-1]>0.5).sum(1)  
            ss1=(score>0.9999)
            ss2=(score>0.6)
            t1=[]
            tx=[]
            dx={}
            dxm={}
            DD={}
            for i in range(px.shape[0]):
                if i not in t1 :
                    tm=[i]
                    tmm=[ql[i].item()]
                    if ss[i]>0:
                        ts=ss1[i,:]>ss1[i,-1]
                        ts1=ss2[i]
                        for  j in range(ts.shape[0]-1):
                            if ts[j]  or (ts1[j] and epyy[i,j]<max(5,0.6*s1[i])):# or (ts1[j] and max(epy[i,j][:2])<2 and epy[i,j,2]<10)):
                                t1.append(q[i,j].item())
                                tm.append(q[i,j].item())
                               
                    dx[i]=tm
              
                    tx.append(i)

        m, n = py.shape[0], py.shape[0]
        py1=py.clone()
        py1[:,2]=py1[:,2]//5
        distmat = torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, py1.float(),py1.float().t())
        qx,q=distmat.topk(min(6,py1.shape[0]),largest=False)
     
        if q.shape[1]<6:
            while q.shape[1]<6:
                q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)   
        qx=qx[:,1:]
        q=q[:,1:]
        eq=q.clone()
  
   
 
            
            
        ey=[]
        ep=[]
        es=[]
        ezs=[]
        #epx=px*px
   
        for jk in range(5):
            ey.append(gf[q[:,jk].unsqueeze(1)])
            t=py[q[:,jk]]-py
            ep.append(torch.abs(t).unsqueeze(1))
        epy=torch.cat(ep,1).float()
        ey=torch.cat(ey,1)
        epyy=torch.sqrt((epy*epy).sum(-1))
    
        for jk in range(5):
            ts1=s2[q[:,jk]]
            es.append(torch.cat([(ts1-s2).unsqueeze(1),(ts1/(s2+1)).unsqueeze(1),
                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
            tzs1=zs2[q[:,jk]]
            ezs.append(torch.cat([zs2.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs2[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs2[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
   
        esp=torch.cat(es,1)
        ezsp=torch.cat(ezs,1)
        with torch.no_grad():
            score=F.sigmoid(EN(gf.unsqueeze(1),ey,epy.float(),esp,ezsp))
        yl=[]
     
        for jk in range(5):
            yl.append((gl[q[:,jk]]).unsqueeze(1))
        yl=torch.cat(yl,1) 
        ss=(score[:,:-1]>0.5).sum(1)  
        ss1=(score>0.9999)
        ss2=(score>0.6)
        t1=[]
        tx=[]
        dx2={}
        dx2m={}
        DD2={}
        dxf={}
        for i in range(py.shape[0]):
            #if i not in t1:
                tm=[i]
                tmm=[i]
                dxf[i]=i
                if ss[i]>0:
                    ts=ss1[i,:]>ss1[i,-1]
                    ts1=ss2[i]
                    for  j in range(ts.shape[0]-1):
                        if ts[j]  or (ts1[j] and epyy[i,j]<max(5,0.6*s2[i])):# or (ts1[j] and max(epy[i,j][:2])<2 and epy[i,j,2]<10)):
                            
                            tm.append(q[i,j].item())
                            dxf[q[i,j].item()]=i
                            if i not in t1:
                                tmm.append(q[i,j].item())
                            t1.append(q[i,j].item())
                                
                          
                dx2[i]=tm
                if i not in t1:
                    dx2m[i]=tmm
                fm={}
                for i in dx2m:
                    for j in dx2m[i]:
                        fm[j]=i
                #for ji in tm:
                 #   dx2[ji]=tm
                tx.append(i)

#######################################################
        # Link cells across different frames
                               
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, px.float(),py.float().t())
        qx,q=distmat.topk(min(5,px.shape[0],py.shape[0]),largest=False)

        if q.shape[1]<5:
            while q.shape[1]<5:
                q=torch.cat([q,q[:,-1].unsqueeze(-1)],1)
                qx=torch.cat([qx,qx[:,-1].unsqueeze(-1)],1)   
      

        ey=[]
        ep=[]
        es=[]
        ezs=[]
        #epx=px*px
   
        for jk in range(5):
            ey.append(gf[q[:,jk].unsqueeze(1)])
            t=py[q[:,jk]]-px
            ep.append(torch.abs(t).unsqueeze(1))
        epy=torch.cat(ep,1).float()
        ey=torch.cat(ey,1)
        epyy=torch.sqrt((epy*epy).sum(-1))
        efy=torch.sqrt((epy*epy)[:,:,:2].sum(-1))
    
        for jk in range(5):
            ts1=s2[q[:,jk]]
            es.append(torch.cat([(ts1-s1).unsqueeze(1),(ts1/(s1+1)).unsqueeze(1),
                                 epyy[:,jk].unsqueeze(1),(ts1-epyy[:,jk]).unsqueeze(1),((ts1-epyy[:,jk])>0).float().unsqueeze(1),(ts1/(epyy[:,jk]+0.0001)).unsqueeze(1)],1).unsqueeze(1))
            tzs1=zs2[q[:,jk]]
            ezs.append(torch.cat([zs1.unsqueeze(1),tzs1.unsqueeze(1),torch.cat([((zs1[:,0]>0.5)*(tzs1[:,1]>0.5)).unsqueeze(1).float(),((zs1[:,0])*(tzs1[:,1])).unsqueeze(1)],1).unsqueeze(1)],2))
   
        esp=torch.cat(es,1)
        ezsp=torch.cat(ezs,1)
     
        with torch.no_grad():
            score=(EX(qf.unsqueeze(1),ey,epy.float(),esp,ezsp))
            sp=score[:,-2:]
            #sp2=score[:,-12:-2]
            score=F.sigmoid(score[:,:-2])
        yl=[]
     
        for jk in range(5):
            yl.append((gl[q[:,jk]]).unsqueeze(1))
           

        #yl.append((ya==0).unsqueeze(1))
        yl=torch.cat(yl,1)
        t1=[]
        tx=[]
        fp=0
        fn=0
        fpd=0
        fnd=0
        IS=0
        c=F.softmax(sp)
        cc=0
        ss1=(score>0.5)
        r={}
        r1s={}
        r2={}
        r2s={}
        r1l={}
        r2l={}
        r1f={}
        r2f={}
        rk={}
        r1su={}
        plq={}
        nn=0
        scorei=score
        for di in dx:
            nn+=1
            lq=min(dx[di])
            if step>0:
                lq=qfm[min(dx[di])]
            if  1:
                rt=[]
                rs=[]
                res=[]
                g=[]
                #lq=nn
                
                su=np.max([s1[i].item() for i in dx[di]])
            
                plq[lq]=torch.cat([p1[i].unsqueeze(0) for i in dx[di]],0).float().mean(0).cpu().numpy()
             
                for i in dx[di]:
                        ts=ss1[i,:]
                      
                        
                        g.append(c[i][1].item())
                        if score[i].argmax(0)<5:#ts.sum()<2:
                            ss=scorei[i].topk(5)[1]
                            zz=dx2[q[i,ss[0]].item()]
                            rt.append(q[i,ss[0]].item())
                            rs.append(scorei[i,ss[0]].item())
                            res.append(epyy[i,ss[0]].item())
                            #st=score[i].argmax().item()
                            for st in ss[1:]:
                                if st==5:break
                                if   q[i,st].item() not in zz :#and q[i,st].item() not in rg:
                                    rt.append(q[i,st].item())
                                    rs.append(scorei[i,st].item())
                                    res.append(epyy[i,st].item())
                                    zz=zz+dx2[q[i,st].item()]
                                                   
                if len(rs)>0:
                    rk=rs.index(np.max(rs))
                    dj={}
                    djn={}
                    for ie in range(len(rt)):
                        dj[rt[ie]]=max(dj.get(rt[ie],0),rs[ie])
                        djn[rt[ie]]=res[ie]
                    nrt=list(dj.keys())
                    nrs=[dj[ie] for ie in nrt]   
                    nres=[djn[ie] for ie in nrt] 
                    
                    #print(g)
                    if 1:
                        dx2[100000]=[]
                        rtx=[]
                        rts=[]
                        zr=[]        
                        zrs=[]
                        zrl=[]
                        rtl=[]
                        zr2=r.get(lq,[100000,100000])
                        zr2s=r1s.get(lq,[0,0])
                        zr2l=r1l.get(lq,[0,0])
                        zr2.append(100000)
                        zr2s.append(0)
                        zr2l.append(0)
                        nrt=nrt+zr2
                        nrs=nrs+zr2s
                        nres=nres+zr2l
                        ss=torch.from_numpy(np.array(nrs)).topk(min(5,len(nrs)))[1]
                        zz=[]
               
                        rtx.append(nrt[ss[0]])
                        rts.append(nrs[ss[0]])
                        rtl.append(nres[ss[0]])
                        
                        zz=zz+dx2[nrt[ss[0]]]                       
                        #st=score[i].argmax().item()
                        for st in ss[1:]:
                            if nrt[st]==100000:break
                            if  nrt[st] not in zz and nres[st]<max(su,5)*6:
                                rtx.append(nrt[st])
                                rts.append(nrs[st])
                                rtl.append(nres[st])
                                zz=zz+dx2[nrt[st]]

                     
                        r[lq]=rtx[:2]#+rtx1
                        r1s[lq]=rts[:2]
                        r1l[lq]=rtl[:2]
                        if r1f.get(lq,0)<np.max(g):r1f[lq]=np.max(g)
                        r1su[lq]=su

        el=[]
        for i in r:
            el.append(r[i])
#            r[i]=[gl[r[i][x]].item() for x in range(min(len(r[i]),1+int(r1f[i]>0.8 or(r1s[i][-1]>0.99 )))) if x!=100000 and (x==0 or ( r1s[i][x]>0.1))]
       
            r[i]=[[fm.get(r[i][x],-1),r1s[i][x]] for x in range(min(len(r[i]),1+int(r1f[i]>0.8 or (r1s[i][-1]>0.5 and r1f[i]>0.5 and np.abs(r1l[i][-1]-r1l[i][0])<1.2*r1su[i] )))) if x!=100000 and ((x==0 and r1s[i][0]>0.1) or (x>0 ))]
   

        zr={}
       
        for i in r:
            for  j in r[i]:
                if zr.get(j[0],0)<j[1]:
                    zr[j[0]]=j[1]
       
        for i in r:
            for  j in r[i]:
                if zr.get(j[0],0)>j[1]:
                    r[i].remove(j)
        r0=r.copy()
######################################## Reconnect broken trajectories
        if step>0:
            qfmv=list(qfm.values())
            for i in r:
                if len(r[i])>0:
                    if i not in qr :
                        if (fq[i][0].item() in qfmv ):
                            if qfm[fq[i][0].item()] in qr and  qfm[fq[i][0].item()] in plq:
                                tf=fq[i][0].item()
                                r0[tf]=r0.get(tf,[])+r[i]
                                r0.pop(i)
                        
                        elif fq[i][1].item() in qfmv:
                            if qfm[fq[i][1].item()] in qr and  qfm[fq[i][1].item()] in plq:
                                tf=fq[i][1].item()
                                r0[tf]=r0.get(tf,[])+r[i]
                                r0.pop(i)
                        
                        elif fq[i][2].item() in qfmv:
                            if qfm[fq[i][2].item()] in qr and  qfm[fq[i][2].item()] in plq:
                                tf=fq[i][2].item()
                                r0[tf]=r0.get(tf,[])+r[i]
                                r0.pop(i)
############################################ Log data                    
        
        #if step==1:ada   
        #fq=eq.copy()
        rr={}
        pr={}
        qr=[]
        for i in r0 :
            if i in plq:
                i0=i
                #i0=int(i.split('-')[1])
                rr[i0]=[j[0] for j in r0[i]]
                qr+=[j[0] for j in r0[i]]
                pr[i0]=plq[i0]+win2#.cpu().numpy()
        #if step==17:asasf
        ZZ[step]=rr
        PZ[step]=pr
        qfm=fm.copy()
        #if step==2:ada
 
#######################################################################

# Integrate data, remove excessively short paths, and output trajectories
tit={}
cidn=1
tidn=1
cid={}
tid={}
t=0
qid={}
pid={}
did={}
temp=PZ[t].copy()
tmp={}
for i in ZZ[t]:
    if (isinstance(ZZ[t][i], list) and len(ZZ[t][i]) > 0) :
        cid[cidn]=[cidn,tidn,t,temp[i][0].item(),temp[i][1].item(),temp[i][2].item(),-1]
        tid[tidn]=tid.get(tidn,[])+[cidn]
        tit[tidn]=tit.get(tidn,[])+[t]
        for j in ZZ[t][i].copy():
            tmp[j]=cidn
        cidn+=1
        tidn+=1
   

t=1
for t in tqdm(range(1,267)):
    ntmp={}
    for i in ZZ[t]:
    
        temp=PZ[t]
        bemp=PZ[t-1]
        pi=tmp.get(i,-1)
        if pi in cid:
            ti=cid[pi][1]
        else:
            tidn+=1
            ti=tidn
        cid[cidn]=[cidn,ti,t,temp[i][0].item(),temp[i][1].item(),temp[i][2].item(),pi]
        tid[ti]=tid.get(ti,[])+[cidn]
        tit[ti]=tit.get(ti,[])+[t]
        for j in ZZ[t][i].copy():
                ntmp[j]=cidn
        cidn+=1
    tmp=ntmp.copy()

plink={}
for i in cid:
    plink[i]=cid[i][-1]
link={}
for i in plink:
    link[plink[i]]=link.get(plink[i],[])+[i]
no=[]
for i in tit:    
   if len(tit[i])<3 and max(tit[i])<260 and max(tid[i]) not in link and min(tit[i])>0:
       no.append(i)
for i in cid.copy():
    if cid[i][1] in no:
        cid.pop(i)
        
D=pd.DataFrame()
lt,l1,l2,l3,l4,l5,l6,l7=[],[],[],[],[],[],[],[]
for i in cid:
        pi=cid[i]
   
        l1.append(pi[0])
        l2.append(pi[1])
        l3.append(pi[2])
        l4.append(pi[3])
        l5.append(pi[4])
        l6.append(pi[5])
        l7.append(pi[6])
#D['t']=lt
D['cellid']=l1
D['trackid']=l2
D['t']=l3
D['x']=l4
D['y']=l5
D['z']=l6
D['parentid']=l7

D.to_csv('track-result.csv',index=False)





    
    
    
    
    
                 
