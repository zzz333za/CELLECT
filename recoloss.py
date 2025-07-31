
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
DK1 = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 2, 1, 0, 0], [0, 1, 2, 3, 2, 1, 0], [
               1, 2, 3, 4, 3, 2, 1], [0, 1, 2, 3, 2, 1, 0], [0, 0, 1, 2, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
DK1 = torch.from_numpy(DK1).float().to(device).reshape([1, 1, 7, 7, 1])

def sort_labels(px,pl1,ql,ee1,ratio=5):
                xy0, xy1, xy2 = torch.where(pl1[0, 0] > 0) 
                xy = torch.cat([xy0.unsqueeze(1), xy1.unsqueeze(
                    1), ratio * xy2.unsqueeze(1)], 1).to(px.device)
                m, n = px.shape[0], xy.shape[0]
                distmat = torch.pow(px[:, :].float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(xy[:, :].float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px[:, :].float(), xy[:, :].float().t())
                qx, q = distmat.sort(1)

                zql = []
                for i in range(q.shape[0]):
                    flag = 0
                    tq = q[i]
                    tqx = qx[i]
                    tx = px[i, 2]
                    tq = tq[tqx < 300]
                    for tnx in range(tq.shape[0]):
                        tn = tq[tnx]
                        ty = xy[tn, 2]
                        flag0 = (torch.abs(tx - ty) == 15).item()
                        flag1 = (torch.abs(tx - ty) == 10).item() * \
                            (tqx[tnx].item() < 200)
                        flag2 = (torch.abs(tx - ty) == 5).item() * \
                            (tqx[tnx].item() < 140)
                        flag3 = (torch.abs(tx - ty) == 0).item() * \
                            (tqx[tnx].item() < 100)
                        flag = flag0 + flag1 + flag2 + flag3
                        if flag:
                            zql.append(
                                pl1[0, 0, xy[tn, 0], xy[tn, 1], xy[tn, 2] // 5].item())
                            break
                    if not flag:
                        zql.append(0)

                lql = (torch.from_numpy(np.array(zql))).cuda()  # *(ql==0)+ql
                pu = torch.where(ql != lql)[0]
                ue = list(np.unique(ee1))[1:]
                if pu.shape[0] > 0:
                    for i in pu:
                        if ql[i] == 0:
                            ql[i] = lql[i]
                        elif lql[i] > 0:
                            if ql[i] in ue:
                                ql[i] = lql[i]

                m, n = px.shape[0], xy.shape[0]
                distmat = torch.pow(px[:, :2].float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(xy[:, :2].float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px[:, :2].float(), xy[:, :2].float().t())
                qx, q = distmat.sort(1)

                zql = []
                for i in range(q.shape[0]):
                    # tn=0
                    flag = 0
                    tq = q[i]
                    tqx = qx[i]
                    tx = px[i, 2]
                    tq = tq[tqx < 16]
                    for tnx in range(tq.shape[0]):
                        tn = tq[tnx]
                        ty = xy[tn, 2]

                        flag2 = (torch.abs(tx - ty) == 5).item()
                        flag3 = (torch.abs(tx - ty) == 0).item()
                        flag = flag2 + flag3
                        if flag:
                            zql.append(
                                pl1[0, 0, xy[tn, 0], xy[tn, 1], xy[tn, 2] // 5].item())
                            break
                    if not flag:
                        zql.append(0)

                lql = (torch.from_numpy(np.array(zql))).cuda() 
                pu = torch.where(ql != lql)[0]
                ue = list(np.unique(ee1))[1:]
                if pu.shape[0] > 0:
                    for i in pu:
                        if ql[i] == 0:
                            ql[i] = lql[i]
                        elif lql[i] > 0:
                            if ql[i] in ue:
                                ql[i] = lql[i]
                return ql


def sort_feature(f,pt,px,q,ss,s1,zss,zs1,n=5):
                ey = []
                ep = []
                es = []
                ezs = []

                for jk in range(n):
                    ey.append(f[q[:, jk].unsqueeze(1)])
                    t = pt[q[:, jk]] - px
                    ep.append(torch.abs(t).unsqueeze(1))
                epy = torch.cat(ep, 1)
                ey = torch.cat(ey, 1)
                epyy = torch.sqrt((epy * epy).sum(-1))

                for jk in range(5):
                    ts1 = ss[q[:, jk]]
                    es.append(torch.cat([(ts1 - s1).unsqueeze(1), (ts1 /
                                             (s1 + 1)).unsqueeze(1), epyy[:, jk].unsqueeze(1), 
                                             (ts1 - epyy[:, jk]).unsqueeze(1), 
                                             ((ts1 - epyy[:, jk]) > 0).float().unsqueeze(1), 
                                             (ts1 / (epyy[:, jk] + 0.0001)).unsqueeze(1)], 1).unsqueeze(1))                       
                    tzs1 = zss[q[:, jk]]
                    ezs.append(torch.cat([zs1.unsqueeze(1),
                                          tzs1.unsqueeze(1),
                                          torch.cat([((zs1[:,0] > 0.5) * (tzs1[:,1] > 0.5)).unsqueeze(1),
                                                     ((zs1[:,0]) * (tzs1[:, 1])).unsqueeze(1)],
                                                    1).unsqueeze(1)],2))

                esp = torch.cat(es, 1)
                ezsp = torch.cat(ezs, 1)
                return ey,epy.float(),esp,ezsp
def ud(x):
    """
    Dilation along Z-axis to expand excluded detection regions.
    """
    x1 = torch.zeros_like(x)
    x2 = torch.zeros_like(x)
    x1[:, :, :, :, :-1] = x[:, :, :, :, 1:]
    x2[:, :, :, :, 1:] = x[:, :, :, :, :-1]
    x0 = x == 0
    xn = (x0 * torch.cat([x1, x2], 1).max(1)[0] - 1)
    xn[xn < 0] = 0
    return x + xn


# -----------------------------------------------------------------------------
# Define search mask
# -----------------------------------------------------------------------------

def kflb(x):

    kn = torch.zeros([x.shape[0], x.shape[1] + 2,
                     x.shape[2] + 2, x.shape[3] + 2])
    kq = torch.zeros([x.shape[0], 27, x.shape[1], x.shape[2], x.shape[3]])
    kn[:, 1:-1, 1:-1, 1:-1] += x.cpu()
    num = 0
    for xc in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                if np.abs(xc) + max(np.abs(y), np.abs(z)) <= 1:
                    kq[:, num] = kn[:, xc + 1:xc + 1 + x.shape[1], y +
                                    1:y + 1 + x.shape[2], z + 1:z + 1 + x.shape[3]]
                    num += 1
    return kq


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an
# -----------------------------------------------------------------------------
# Loss function for training center point extraction
# -----------------------------------------------------------------------------
weights = torch.tensor([1.0, 32.0]).cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weights)
def crloss(a, b):
    tl1 = 0  # criterion5(a,(b).long())
    for ti in range(1, a.shape[1]):
        tl1 += criterion(torch.cat([a[:, :1],
                         a[:, ti:ti + 1]], 1), (b >= 1).long())
    for ti in range(2, a.shape[1]):
        tl1 += criterion(torch.cat([a[:,
                                      :ti].max(1)[0].unsqueeze(1),
                                    a[:,
                                      ti:ti + 1]],
                                   1),
                         (b >= 2).long()) * ((b == 0) + (b >= 2))
    for ti in range(3, a.shape[1]):
        tl1 += criterion(torch.cat([a[:,
                                      :ti].max(1)[0].unsqueeze(1),
                                    a[:,
                                      ti:ti + 1]],
                                   1),
                         (b >= 3).long()) * ((b == 0) + (b >= 3))
    for ti in range(4, a.shape[1]):
        tl1 += criterion(torch.cat([a[:,
                                      :4].max(1)[0].unsqueeze(1),
                                    a[:,
                                      ti:ti + 1]],
                                   1),
                         (b >= 4).long()) * ((b == 0) + (b >= 4))
    return tl1

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
########## U-Net feature extraction process for the 256*256*32 patch image 
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

########## U-Net feature extraction process for the entire image (all patches)
def run_unet_on_patches(feature_extract_net, volume_tensor, label_tensor):
    """
    Sliding-window inference over the full volume via patches.

    Parameters
    ----------
    feature_extract_net : nn.Module
        Pretrained 3D U‐Net model.
    volume_tensor : torch.Tensor, shape (B, 2, H, W, Z)
        Full volume input tensor (batch size B).
    label_tensor : torch.Tensor
        Full volume ground-truth labels.

    Returns
    -------
    p, f, zs, s, lb, uout : tuple
        - p: Tensor[M,4] all coordinates
        - f: Tensor[M,C] all feature vectors
        - zs: Tensor[M,2] all probabilities
        - s: Tensor[M] all sizes
        - lb: Tensor[M] all labels
        - uout: Tensor[H,W,Z] aggregated UNet prediction
    """
    x = volume_tensor
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
                    p1,f1,s1,zs1,lb1,ar,uo=  vsup256(feature_extract_net, x[:,:,v1:v2,v3:v4,v5:v6],label_tensor[:,v1:v2,v3:v4,v5:v6])  
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
def validation_simple(U, EN, EX, inputs, in2, in3 , p1, p2, labels, la2, s1, s2, s4, PA, step):
            tm=2
            weights = torch.tensor([1.0, 32.0]).cuda()
            criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weights)
            weights2 = torch.tensor([1.0, 32.0, 1.0]).cuda()
            criterion3 = torch.nn.CrossEntropyLoss(reduction='none', weight=weights2)
            weights5 = torch.tensor([1.0, 32.0, 32, 32, 32]).cuda()
            criterion5 = torch.nn.CrossEntropyLoss(reduction='none', weight=weights5)
            weights6 = torch.tensor([1.0, 1]).cuda()
            criterion6 = torch.nn.CrossEntropyLoss(reduction='none', weight=weights6)
            criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
            rkloss = torch.nn.MarginRankingLoss(margin=0.3)
            Tloss = TripletLoss(0.3)
            l1loss = torch.nn.L1Loss(reduction='none')
            mse = torch.nn.MSELoss()
            bce = torch.nn.BCELoss(reduction='none')
            pl1=p1.clone()
            pl2=p2.clone()
            ee1 = list((pl1[0, 0][pl1[0, 0] != 0]).flatten().cpu().numpy())
            ee2 = list((pl2[0, 0][pl2[0, 0] != 0]).flatten().cpu().numpy())
            lx2 = torch.unique(p1[0, 0]).shape[0]
            with torch.no_grad():
                Uout, uo, fo, zs1, size1 = U(PA(torch.cat([inputs, in2], 1)))
                Uout2, uo2, fo2, zs2, size2 = U(PA(torch.cat([in2, in3], 1)))
               
                p1[:, 0] = F.conv3d((p1[:, 0] > 0).float().unsqueeze(
                    1), DK1, padding=(tm + 1, tm + 1, 0))
                p2[:, 0] = F.conv3d((p2[:, 0] > 0).float().unsqueeze(
                    1), DK1, padding=(tm + 1, tm + 1, 0))
                for i in range(3):
                    p1 = ud(p1)
                    p2 = ud(p2)
                wp1 = labels > 0
                zp1 = wp1 * (p1[:, 0])
                wp2 = la2 > 0
                zp2 = wp2 * (p2[:, 0])
                rloss3 = 0
                kloss = []
                Uf = fo.transpose(0, 1)
                Uf2 = fo2.transpose(0, 1)
            
                uc = ((kflb(uo[:, :].max(1)[0]).cuda().max(1)[
                           0] == uo[:, 4])) * (Uout.argmax(1) == 1)

                u, x, y, z = torch.where(uc.to(device, dtype=torch.float))
                lx = torch.unique(labels).shape[0]
                lx1 = torch.unique(labels * uc).shape[0]
                lx3 = list(torch.unique(labels * uc).cpu().numpy())
                lx3 = [i for i in lx3 if i in np.unique(ee1)]
                f1 = Uf[:, u, x, y, z].transpose(0, 1)
                zb = s1[u, x, y, z] == 1
                s1 = size1[u, 0, x, y, z]
                zs1 = F.sigmoid(zs1[u, :, x, y, z])

                lb1 = labels[u, x, y, z]
      

                uc = ((kflb(uo2[:, :].max(1)[0]).cuda().max(1)[
                           0] == uo2[:, 4])) * (Uout2.argmax(1) == 1)
                nloss1 = crloss(uo, (zp1).long())
                nloss2 = crloss((uo2), (zp2).long())
          
                Uloss = torch.mean(nloss1) + torch.mean(nloss2) 

            if u.shape[0] > 0:

                u2, x2, y2, z2 = torch.where(uc.to(device, dtype=torch.float))
                f2 = Uf2[:, u2, x2, y2, z2].transpose(0, 1)
                s2 = size2[u2, 0, x2, y2, z2]
                zs2 = F.sigmoid(zs2[u2, :, x2, y2, z2])
                lb2 = la2[u2, x2, y2, z2]
                p1 = torch.cat([u.unsqueeze(1), x.unsqueeze(
                    1), y.unsqueeze(1), z.unsqueeze(1)], 1)
                p2 = torch.cat([u2.unsqueeze(1), x2.unsqueeze(
                    1), y2.unsqueeze(1), z2.unsqueeze(1)], 1)

                px = p1.float()
                py = p2.float()
                px[:, 0] = px[:, 0] * 100
                py[:, 0] = py[:, 0] * 100
                m, n = px.shape[0], py.shape[0]


                p1 = p1[:, 1:]
                p2 = p2[:, 1:]
                p1[:, 2] = p1[:, 2] * 5
                p2[:, 2] = p2[:, 2] * 5
                px = p1.float()
                py = p2.float()

                qf, ql, px, gf, gl, py = f1, lb1, p1, f2, lb2, p2
                if qf.shape[0] > 5 and gf.shape[0] > 5:

                    # -----------------------------------------------------------------------------
                    # frame 1

                    ql = sort_labels(px, pl1, ql, ee1, ratio=5)

                    # -----------------------------------------------------------------------------
                    # frame 2

                    gl = sort_labels(py, pl2, gl, ee2, ratio=5)

                    # -----------------------------------------------------------------------------
                    # frame 1
                    m, n = px.shape[0], px.shape[0]
                    distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    distmat.addmm_(1, -2, px.float(), px.float().t())
                    qx, q = distmat.topk(6, largest=False)
                    qx = qx[:, 1:]
                    q = q[:, 1:]

                    ey, epy, esp, ezsp = sort_feature(qf, px, px, q, s1, s1, zs1, zs1, n=5)
                    with torch.no_grad():
                        score = EN(qf.unsqueeze(1), ey, epy.float(), esp, ezsp)
                    yl = []
                    ya = 0
                    for jk in range(5):
                        yl.append((ql[q[:, jk]] == ql).unsqueeze(1))
                        ya += (ql[q[:, jk]] == ql).float()

                    yl.append((ya == 0).unsqueeze(1))
                    yl = torch.cat(yl, 1)
                    yls = yl[:, :5].float().sum(1)
                    ylmax = yl.float().argmax(1)
                    ylmin = yl[:, :5].float().argmin(1)
                    score = score[ql != 0]
                    yls = yls[ql != 0]
                    yl = yl[ql != 0]

                    tkloss = torch.mean(bce(F.sigmoid(score), yl.float()))
                    if step % 10 == 0:
                        print(
                            'selfex:acc0:',
                            ((score.cpu().argmax(1) < 5) == (
                                    yls > 0).cpu()).sum().item(),
                            score.shape[0],
                            labels.unique().shape,
                            la2.unique().shape)
                        print(
                            'selfex:acc:', ((F.sigmoid(score).cpu().float() > 0.5) == (
                                yl.cpu().float())).sum().item(), ((score.cpu().float() > 0.5) == (
                                yl.cpu().float())).sum().item() / (
                                                                         score.shape[1] * score.shape[0]))
                    # -----------------------------------------------------------------------------
                    # inter-frame
                    m, n = qf.shape[0], gf.shape[0]
                    distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    distmat.addmm_(1, -2, px.float(), py.float().t())
                    qx, q = distmat.topk(5, largest=False)

                    ey, epy, esp, ezsp = sort_feature(gf, py, px, q, s2, s1, zs2, zs1, n=5)
                    with torch.no_grad():
                        score = EX(qf.unsqueeze(1), ey, epy.float(), esp, ezsp)
                    sp = score[:, -2:]
                    score = score[:, :-2]
                    yl = []
                    ya = 0
                    for jk in range(5):
                        yl.append((gl[q[:, jk]] == ql).unsqueeze(1))
                        ya += (gl[q[:, jk]] == ql).float()
                    zcs = criterion(sp, zb.long())
                    yl.append((ya == 0).unsqueeze(1))
                    yl = torch.cat(yl, 1)
                    yls = yl[:, :5].float().sum(1)
                    ylmax = yl.float().argmax(1)
                    ylmin = yl[:, :5].float().argmin(1)
                    score = score[ql != 0]
                    yls = yls[ql != 0]
                    yl = yl[ql != 0]
                    cl = bce(F.sigmoid(score), yl.float())
                    tkloss += torch.mean(cl) + torch.mean(zcs)
                    zz = sp.argmax(1)
                    if step % 10 == 0 or lx2 > 400:
                        print(
                            'ex:acc0:',
                            ((score.cpu().argmax(1) < 5) == (
                                    yls > 0).cpu()).sum().item(),
                            score.shape[0],
                            labels.unique().shape,
                            la2.unique().shape)
                        print(
                            'ex:acc:', ((F.sigmoid(score).cpu().float() > 0.5) == (
                                yl.cpu().float())).sum().item(), ((score.cpu().float() > 0.5) == (
                                yl.cpu().float())).sum().item() / (
                                                                         score.shape[1] * score.shape[0]))
                        print('seg:', lx, lx1, lx2, len(lx3))
                        print(
                            'zacc:',
                            ((zb == zz).sum() / zb.shape[0]).item(),
                            zb.sum().item(),
                            zz.sum().item(),
                            (zb * zz).sum().item())

                    tcloss = 0

                    kloss.append((tcloss + tkloss).unsqueeze(-1))
                lb1 = ql
                lb2 = gl
                til = list((lb1).cpu().numpy())
                z1 = pd.value_counts(til).to_dict()
                til = list((lb2).cpu().numpy())

                z2 = pd.value_counts(til).to_dict()
                cb1 = lb1.cpu().numpy().astype(int)
                cb2 = lb2.cpu().numpy().astype(int)

                # -----------------------------------------------------------------------------

                tt = []
                tt1 = []
                lt = []
                for ix in range(len(lb1)):
                    if cb1[ix] != 0 and cb1[ix] not in lt:
                        ta = z2.get(cb1[ix], 0)

                        if ta >= 1:
                            tt.append(ix)
                        if ta >= 2:
                            tt1.append(ix)
                        lt.append(cb1[ix])
                tt1 = np.array(tt1)
                f1m = f1[tt1]
                p1m = p1[tt1]
                lb1m = lb1[tt1]
                tt = np.array(tt)
                f1 = f1[tt]
                p1 = p1[tt]
                lb1 = lb1[tt]

                # '

                tt = []
                lt = []
                tt1 = []
                for ix in range(len(lb2)):
                    if cb2[ix] != 0 and z1.get(cb2[ix], 0) > 0:
                        ta = z2.get(cb2[ix], 0)
                        kt = lt.count(cb2[ix])
                        if kt == 0 and ta >= 1:
                            tt.append(ix)
                            lt.append(cb2[ix])
                        if kt == 1 and ta >= 2:
                            tt1.append(ix)
                            lt.append(cb2[ix])
                tt1 = np.array(tt1)
                f2m = f2[tt1]
                p2m = p2[tt1]
                lb2m = lb2[tt1]
                tt = np.array(tt)
                f2 = f2[tt]
                p2 = p2[tt]
                lb2 = lb2[tt]
                tt = []
                ZL = [[[f1, f2], [lb1, lb2]], [[f1m, f2m], [lb1m, lb2m]]]

                for inx in ZL:

                    Ff, L = inx
                    Ff = torch.cat(Ff, 0)
                    L1 = torch.cat(L, 0)
                    if Ff.shape[0] > 4:
                        lt = Tloss(Ff, L1)[0]
                        if lt.item() == lt.item():
                            rloss3 += lt
            return Uloss, rloss3, kloss