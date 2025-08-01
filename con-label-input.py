# -*- coding: utf-8 -*-


from PIL import Image

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from skimage import data, util,draw
parser = argparse.ArgumentParser(description="Training script for the model")

parser.add_argument('--data_dir', type=str, required=True, help="Path to the training data directory")
parser.add_argument('--out_dir', type=str, required=True, help="Path to the output data directory")
parser.add_argument('--num', type=str, required=True, help="Dataset id")
parser.add_argument('--start', type=int, default=0, help="start frame (default: 0)")
parser.add_argument('--end', type=int, default=275, help="end frame (default: 275)")






# This program converts the sparse annotated position points and trajectory information
# into a combination of a matrix and mask format.
args = parser.parse_args()
datap=args.data_dir
outp=args.out_dir
st=int(args.start)
ed=int(args.end)
num=[args.num]

if not os.path.exists(outp+'/ls'+str(num[0])+'/'):
    os.mkdir(outp+'/ls'+str(num[0])+'/')

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


def fill(n,x,y,z,v=1,s=4,r=1):
    rr,cc=draw.ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
    z1=int(max(0,z-1-s//5))
    z2=int(min(41,z+2+s//5))

    if z1==z2:
        n[rr,cc,z1]=v
    else:
        n[rr,cc,z1:z2]=v
    return n
def fill1(n,x,y,z,v=1):

    n[int(x),int(y),int(z)]=v
    return n
for op in num:
    op=str(op)
    l=os.listdir(datap+'/mskcc_confocal_s'+op+'/images/')
    l=[i for i in l if 'tif' in i]
    
    D={}
    for i in tqdm(l):
        if 'tif' in i:         
            
            num=int(i.split('_')[-1].split('.')[0][1:])
            #a=Image.open('D:/mskcc-confocal/mskcc_confocal_s1/images/'+i)
            
            D[num]=datap+'/mskcc_confocal_s'+op+'/images/'+i
            
    #d1=d[d['Time']==1]
    #d2=d[d['Time']==2]
    
    d1=pd.read_table(datap+'/mskcc_confocal_s'+op+'/tracks/tracks.txt')#,skiprows=3)
    d2=pd.read_table(datap+'/mskcc_confocal_s'+op+'/tracks/tracks_polar_bodies.txt')#,skiprows=3)
 
    for t in tqdm(range(st,ed)):
        
        c1=tim(Image.open(D[t]))
 
        d=d1.loc[d1['t']==t]
        dx=d2.loc[d2['t']==t]
        dxt=d1.loc[d1['t']==t+1]
        dxtt=d2.loc[d2['t']==t+1]
        im24=np.zeros(c1.shape)
        im23=np.zeros(c1.shape)
        im22=np.zeros(c1.shape)
        im21=np.zeros(c1.shape)
        im31=np.zeros(c1.shape)
        im32=np.zeros(c1.shape)
        im6=np.zeros(c1.shape)
        im5=np.zeros(c1.shape)
        im4=np.zeros(c1.shape)
        im3=np.zeros(c1.shape)
        im2=np.zeros(c1.shape)
        im1=np.zeros(c1.shape)
        for i in (range(d.shape[0])):
    
            im1[:,:,:]=fill1(im1[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['cell_id'].values[i]+1))
            im5[:,:,:]=fill(im5[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['div_state'].values[i]),int(d['radius'].values[i])//1,int(d['radius'].values[i])//5)
            im21[:,:,:]=fill(im21[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['cell_id'].values[i]+1),int(d['radius'].values[i])//1,int(d['radius'].values[i])//5)
            im31[:,:,:]=fill(im31[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,float(d['radius'].values[i]),int(d['radius'].values[i])//1,int(d['radius'].values[i])//5)
        for i in (range(d.shape[0])):
    
            #im1[:,:,:]=fill1(im1[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['cell_id'].values[i]+1))
            im5[:,:,:]=fill(im5[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['div_state'].values[i]),int(d['radius'].values[i])//2,int(d['radius'].values[i])//5)
            im21[:,:,:]=fill(im21[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,int(d['cell_id'].values[i]+1),int(d['radius'].values[i])//2,int(d['radius'].values[i])//5)
            im31[:,:,:]=fill(im31[:,:,:],d['y'].values[i],d['x'].values[i],d['z'].values[i]//5,float(d['radius'].values[i]),int(d['radius'].values[i])//2,int(d['radius'].values[i])//5)

        for i in (range(dx.shape[0])):
            
            im23[:,:,:]=fill(im23[:,:,:],dx['y'].values[i],dx['x'].values[i],dx['z'].values[i]//5,int(dx['cell_id'].values[i]+1),int(dx['radius'].values[i])//1,int(dx['radius'].values[i])//5)
           
            im3[:,:,:]=fill1(im3[:,:,:],dx['y'].values[i],dx['x'].values[i],dx['z'].values[i]//5,int(dx['cell_id'].values[i]+1))
            im31[:,:,:]=fill(im31[:,:,:],dx['y'].values[i],dx['x'].values[i],dx['z'].values[i]//5,float(dx['radius'].values[i]),int(dx['radius'].values[i])//1,int(dx['radius'].values[i])//5)

        for i in (range(dxtt.shape[0])):
            im24[:,:,:]=fill(im24[:,:,:],dxtt['y'].values[i],dxtt['x'].values[i],dxtt['z'].values[i]//5,int(dxtt['parent_id'].values[i]+1),int(dxtt['radius'].values[i])//1,int(dxtt['radius'].values[i])//5)

            im4[:,:,:]=fill1(im4[:,:,:],dxtt['y'].values[i],dxtt['x'].values[i],dxtt['z'].values[i]//5,int(dxtt['parent_id'].values[i]+1))
            im32[:,:,:]=fill(im32[:,:,:],dxtt['y'].values[i],dxtt['x'].values[i],dxtt['z'].values[i]//5,float(dxtt['radius'].values[i]),int(dxtt['radius'].values[i])//1,int(dxtt['radius'].values[i])//5)


        for i in (range(dxt.shape[0])):
            im2[:,:,:]=fill1(im2[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['parent_id'].values[i]+1))
 
            im6[:,:,:]=fill(im6[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['div_state'].values[i]),int(dxt['radius'].values[i])//1,int(dxt['radius'].values[i])//5)
    
    
    
    
            #im2[:,:,:]=fill(im2[:,:,:],d['Position Y'].values[i],d['Position X'].values[i],d['Position Z'].values[i],int(d['TrackID'].values[i]%1e9),3)
            im22[:,:,:]=fill(im22[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['parent_id'].values[i]+1),int(dxt['radius'].values[i])//1,int(dxt['radius'].values[i])//5)
            im32[:,:,:]=fill(im32[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,float(dxt['radius'].values[i]),int(dxt['radius'].values[i])//1,int(dxt['radius'].values[i])//5)
        for i in (range(dxt.shape[0])):
            #im2[:,:,:]=fill1(im2[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['parent_id'].values[i]+1))
 
            im6[:,:,:]=fill(im6[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['div_state'].values[i]),int(dxt['radius'].values[i])//2,int(dxt['radius'].values[i])//5)
    
    
    
    
            #im2[:,:,:]=fill(im2[:,:,:],d['Position Y'].values[i],d['Position X'].values[i],d['Position Z'].values[i],int(d['TrackID'].values[i]%1e9),3)
            im22[:,:,:]=fill(im22[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,int(dxt['parent_id'].values[i]+1),int(dxt['radius'].values[i])//2,int(dxt['radius'].values[i])//5)
            im32[:,:,:]=fill(im32[:,:,:],dxt['y'].values[i],dxt['x'].values[i],dxt['z'].values[i]//5,float(dxt['radius'].values[i]),int(dxt['radius'].values[i])//2,int(dxt['radius'].values[i])//5)
                
        np.save(outp+'/ls'+op+'/'+str(t)+'-k1-3d-1-imaris.npy',im1)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k2-3d-1-imaris.npy',im2)
        # np.save(outp+'/ls'+op+'/'+str(t)+'-k3--3d-1-imaris.npy',im3)
        # np.save(outp+'/ls'+op+'/'+str(t)+'-k4-3d-1-imaris.npy',im4)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k5-3d-1-imaris.npy',im5)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k6-3d-1-imaris.npy',im6)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k1-3d-imaris.npy',im21)    
        np.save(outp+'/ls'+op+'/'+str(t)+'-k2-3d-imaris.npy',im22)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k3-3d-imaris.npy',im23)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k4-3d-imaris.npy',im24)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k31-imaris.npy',im31)
        np.save(outp+'/ls'+op+'/'+str(t)+'-k32-imaris.npy',im32)
