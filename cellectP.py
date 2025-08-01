


import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm  import tqdm
import tifffile
from unetext3Dn_con7s import UNet3D

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
        self.outc=nn.Linear(self.in_channels+self.n_classes, 2)
        self.d1= nn.Dropout(p=0.1)
        self.d2= nn.Dropout(p=0.1)
        self.s=nn.Sigmoid()
    def forward(self, x,y,p,s,zs):
      
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
        xx=torch.sqrt((x*x).sum(2))
        yy=torch.sqrt((y*y).sum(2))
        xy=x*y
        xy2=(xx*yy).unsqueeze(-1)
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
        ui=self.bn7(self.lrelu(self.fc(((nx))) ))
        nx=self.lrelu(self.fc2(ui))
        f=(self.out(nx))     
        return f

############################################### Define search masks


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


def ellipse(r, c, r_radius, c_radius, shape=None):


    y, x = np.ogrid[-r_radius:r_radius+1, -c_radius:c_radius+1]

    mask = (y**2)/(r_radius**2) + (x**2)/(c_radius**2) <= 1

    rr_local, cc_local = np.nonzero(mask)

    rr = rr_local + (r - r_radius)
    cc = cc_local + (c - c_radius)

    if shape is not None:
        H, W = shape[:2]
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr = rr[valid]
        cc = cc[valid]

    return rr, cc

def fill(n,x,y,z,v=1,s=4,r=5):
    rr,cc=ellipse(int(x),int(y), s, s)
    ir=(rr>0)*(rr<n.shape[0])
    ic=(cc>0)*(cc<n.shape[1])
    ii=ir*ic
    rr=rr[ii]
    cc=cc[ii]
    z1=int(max(0,z-1-s//r))
    z2=int(min(n.shape[2],z+2+s//r))

    if z1==z2:
        n[rr,cc,z1]=v
    else:
        n[rr,cc,z1:z2]=v
    return n

# Data preprocessing


def ud(x):
    x1=torch.zeros_like(x)
    x2=torch.zeros_like(x)
    x1[:,:,:,:,:-1]=x[:,:,:,:,1:]
    x2[:,:,:,:,1:]=x[:,:,:,:,:-1]
    x0=x==0
    xn=(x0*torch.cat([x1,x2],1).max(1)[0]-1)    
    xn[xn<0]=0
    return x+xn
DK1=np.array([[0,0,1,1,1,0,0],[0,1,1,2,1,1,0],[1,1,2,3,2,1,1],[1,2,3,4,3,2,1],[1,1,2,3,2,1,1],[0,1,1,2,1,1,0],[0,0,1,1,1,0,0]])
DK1=torch.from_numpy(DK1).float().reshape([1,1,7,7,1])#.repeat(1,1,1,1,3)
######################## Test each patch using U-Net
def feature_extract_patch(U,x,la):
    device = x.device
    Uout,uo,foz,zs1,size1 =  U(x)
    Uf=foz.transpose(0,1)
    uo[:, :5] = F.conv3d(
        uo[:, :5].float(),
        DK1.expand(5, 1, *DK1.shape[2:]).to(device),  # shape [5,1,7,7,1]
        padding=(3,3,0),
        groups=5
    )
    # uu=(Uout.argmax(1)==1)
    # uxo=uo[:,4]
    #uc=(F.max_pool3d(uxo, kernel_size=3, stride=1, padding=2)==uxo)*uu
    uc=((F.max_pool3d(uo[:,:].max(1)[0], kernel_size=3, stride=1, padding=1)==uo[:,4]))*(kflb(Uout.argmax(1)==0).cuda().sum(1)==0)
    #uc=((kflb(uo[:,:].max(1)[0]).cuda().max(1)[0]==uo[:,4]))*(Uout.argmax(1)>0)
    uc[:,:3]=0
    uc[:,-3:]=0
    uc[:,:,:3]=0
    uc[:,:,-3:]=0
    uc[:,:,:,:3]=0
    uc[:,:,:,-2:]=0    
    ar=F.conv3d((uc>0).float().unsqueeze(1), DK1.to(device), padding=(3,3,0))  
    #ar=F.conv3d((ar>0).float(), DK1.to(device), padding=(3,3,0))
    for i in range(1):
            ar=ud(ar)   
    u,x,y,z=torch.where(uc.to(device, dtype=torch.float))
    f1=Uf[:,u,x,y,z].transpose(0,1)
    s1=size1[u,0,x,y,z]
    zs1=F.sigmoid(zs1[u,:,x,y,z])
    lb1=la[u,x,y,z]  
    p1=torch.cat([u.unsqueeze(1),x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],1)
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
def feature_extract(U,x,l):
    u,c,pl1,pl2,pl3=x.shape
    ax=pl3
    #print(ax)
    if pl3<32:
        x=torch.cat([x,x.min()*torch.ones([u,c,pl1,pl2,32-pl3]).to(x.device)],-1)
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
          
                with torch.no_grad():
                    p1,f1,s1,zs1,lb1,ar,uo=  feature_extract_patch(U,x[:,:,v1:v2,v3:v4,v5:v6],l[:,v1:v2,v3:v4,v5:v6])  
                uout[v1+2:v2-2,v3+2:v4-2,v5:v6]+=(uo.argmax(1).squeeze().cpu()[2:-2,2:-2]==1).float()
                    
                ar[ar>1]=1
                ar=ar.squeeze().cpu()

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
      
           
                    p=torch.cat([p,p1],0)
                    f=torch.cat([f,f1],0)
                    s=torch.cat([s,s1],0)
                    zs=torch.cat([zs,zs1],0)
              
                else:
                    p=p1
                    f=f1
                    s=s1
                    zs=zs1
            
                ku[v1:v2,v3:v4,v5:v6]+=ar
    fn=p[:,-1]<ax-1
    p=p[fn]
    f=f[fn]
    s=s[fn]
    zs=zs[fn]
    return p,f,zs,s,s,uout[:,:,:ax]
################################# Load and test the final frame to determine the processing scope

def track(inputs, in2, in3, zratio, PA, U, EX, EN, div = 1):
    device = inputs.device
    o1={}
    o2={}
    with torch.no_grad():
                p1,f1,zs1,s1,lb1,uout =  feature_extract(U,(PA(torch.cat([inputs,in2],1))),inputs[:,0]*0)

                p2,f2,zs2,s2,lb2,uo2 =  feature_extract(U,(PA(torch.cat([in2,in3],1))),in2[:,0]*0) 
    p1=p1[:,1:]
    p2=p2[:,1:]
      
    tq=np.zeros(inputs.squeeze().shape)     
    tq2=np.zeros(inputs.squeeze().shape)   
    u=[]
    for i in range(p1.shape[0]):
        x,y,z=p1[i]
        x=int(x)
        y=int(y)
        z=int(z)
        s=int(s1[i].item()*1.2)
        if tq[x,y,z]==0:
            tq=fill(tq,x,y,z,s=max(3,s),r=zratio,v=i+1)
            u.append(i)
    u=np.array(u)
    p1=p1[u]
    f1=f1[u]
    zs1=zs1[u]
    s1=s1[u]
    lb1=lb1[u]
    u=[]
    for i in range(p2.shape[0]):
            x,y,z=p2[i]
            x=int(x)
            y=int(y)
            z=int(z)
            s=int(s2[i].item()*1.2)
            if tq2[x,y,z]==0:
                tq2=fill(tq2,x,y,z,s=max(3,s),r=zratio,v=i+1)
                u.append(i)
    u=np.array(u)
    p2=p2[u]
    f2=f2[u]
    zs2=zs2[u]
    s2=s2[u]
    lb2=lb2[u]
    p1[:,2]=p1[:,2]*1
    p2[:,2]=p2[:,2]*1
    
    qf,ql,px,gf,gl,py=f1,lb1,p1,f2,lb2,p2
    if p1.shape[0]>0 and p2.shape[0]>0:
    ##############################################################
        # Use the MLP to find points belonging to the same cell within the same frame and exclude them
        m, n = px.shape[0], px.shape[0]
        px1=px.clone()
        px1[:,2]=px1[:,2]//zratio
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
                        if ts[j]  or (ts1[j] and epyy[i,j]<max(5,0.6*s1[i])):
                            t1.append(q[i,j].item())
                            tm.append(q[i,j].item())   
                dx[i]=tm              
                tx.append(i)
        o1['intra-score']=score
    ########################################################################
        m, n = py.shape[0], py.shape[0]
        py1=py.clone()
        py1[:,2]=py1[:,2]//zratio
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
                tm=[i]
                tmm=[i]
                dxf[i]=i
                if ss[i]>0:
                    ts=ss1[i,:]>ss1[i,-1]
                    ts1=ss2[i]
                    for  j in range(ts.shape[0]-1):
                        if ts[j]  or (ts1[j] and epyy[i,j]<max(5,0.6*s2[i])):
                            
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
                tx.append(i)
        o2['intra-score']=score
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
            score=F.sigmoid(score[:,:-2])
        yl=[]  
        for jk in range(5):
            yl.append((gl[q[:,jk]]).unsqueeze(1))
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
    
            if  1:
                rt=[]
                rs=[]
                res=[]
                g=[]
                su=np.max([s1[i].item() for i in dx[di]])
            
                plq[lq]=torch.cat([p1[i].unsqueeze(0) for i in dx[di]],0).float().mean(0).cpu().numpy()
             
                for i in dx[di]:
                        ts=ss1[i,:]
                        g.append(c[i][1].item())
                        if score[i].argmax(0)<5:
                            ss=scorei[i].topk(5)[1]
                            zz=dx2[q[i,ss[0]].item()]
                            rt.append(q[i,ss[0]].item())
                            rs.append(scorei[i,ss[0]].item())
                            res.append(epyy[i,ss[0]].item())
                            for st in ss[1:]:
                                if st==5:break
                                if   q[i,st].item() not in zz :
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
                
                        for st in ss[1:]:
                            if nrt[st]==100000:break
                            if  nrt[st] not in zz and nres[st]<max(su,5)*6:
                                rtx.append(nrt[st])
                                rts.append(nrs[st])
                                rtl.append(nres[st])
                                zz=zz+dx2[nrt[st]]
                        r[lq]=rtx[:2]
                        r1s[lq]=rts[:2]
                        r1l[lq]=rtl[:2]
                        if r1f.get(lq,0)<np.max(g):r1f[lq]=np.max(g)
                        r1su[lq]=su
    
        el=[]
        for i in r:
            el.append(r[i])
            if div==0:
                r[i]=[[fm.get(r[i][x],-1),r1s[i][x]] for x in range(1) if x!=100000 and ((x==0 and r1s[i][0]>0.1) or (x>0 ))]
            else: 
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
        for i in r:
            if len(r.get(i,-1))==-1:
                r.pop(i)
            else:
                r[i]=[j[0] for j in r[i]]
        Z={}
        tq=np.zeros(inputs.squeeze().shape)     
        tq2=np.zeros(inputs.squeeze().shape)  
        u1=list(r.keys())
        for i in range(p1.shape[0]):
            if i in u1:
                x,y,z=p1[i]
                x=int(x)
                y=int(y)
                z=int(z)
                s=int(s1[i].item()*1.2)

                tq=fill(tq,x,y,z,s=max(3,s),r=zratio,v=i+1)
                for j in r[i]:
                    x,y,z=p2[j]
                    x=int(x)
                    y=int(y)
                    z=int(z)
                    s=int(s2[j].item()*1.2)
                    tq2=fill(tq2,x,y,z,s=max(3,s),r=zratio,v=j+1)

        o1['pos']=p1
        o1['feature']=f1
        o1['size']=s1
        o1['div']=zs1
        o1['seg']=uout
        o1['seg_mask']=tq*(uout.cpu().numpy()>0)#.float()
        o1['group']=dx
        o2['pos']=p2
        o2['feature']=f2
        o2['size']=s2
        o2['div']=zs2
        o2['seg']=uo2
        o2['seg_mask']=tq2*(uo2.cpu().numpy()>0)
        o2['group']=dx2  
        Z['frame1']=o1
        Z['frame2']=o2
        Z['inter-score']=score
        Z['link']=r
        return Z




    
    
    
    
    
                 
