
import os
import argparse
import random
from random import randint
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import tifffile
from tqdm import tqdm
from recoloss import CrossEntropyLabelSmooth, TripletLoss, crloss, ud, sort_feature, sort_labels
from recoloss import kflb as search_mask
from unetext3Dn_con7s import UNet3D, EXNet
# Ensure KMP duplicates do not error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Argument parser setup
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Training script for the model")

parser.add_argument('--data_dir', type=str, required=True,
                    help="Path to the training data directory")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Path to the output data directory")
parser.add_argument(
    '--cpu',
    action='store_true',
    help="Use CPU instead of GPU if this flag is set.")

parser.add_argument(
    '--model1_dir',
    type=str,
    required=True,
    help="Path to the Unet model")
parser.add_argument(
    '--model2_dir',
    type=str,
    required=True,
    help="Path to MLP model1")
parser.add_argument(
    '--model3_dir',
    type=str,
    required=True,
    help="Path to MLP model2")


parser.add_argument('--test', type=str, required=True, help="Test data id")


args = parser.parse_args()
datap = args.data_dir
outp = args.out_dir
mp1 = args.model1_dir
mp2 = args.model2_dir
mp3 = args.model3_dir
OP = int(args.test)
# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
op = str(OP)
vl = os.listdir(datap + '/mskcc_confocal_s' + op + '/images/')
vl = [i for i in vl if 'tif' in i]

vD = {}
for i in tqdm(vl):
    if 'tif' in i:

        num = int(i.split('_')[-1].split('.')[0][1:])
        if num < 273:

            vD[num] = datap + '/mskcc_confocal_s' + op + '/images/' + i


vd1 = pd.read_table(datap + '/mskcc_confocal_s' + op +
                    '/tracks/tracks.txt')  
vd2 = pd.read_table(datap + '/mskcc_confocal_s' + op +
                    '/tracks/tracks_polar_bodies.txt') 
# -----------------------------------------------------------------------------
# Configure the data loader
# -----------------------------------------------------------------------------

class VDataset(Dataset):

    def __init__(self, data, le):
        self.data = data

    def __len__(self):

        return len(self.data) - 5

    def __getitem__(self, i):

        j = i
        g = vD[j]
        o = 1
        g2 = vD[j + o]
        g3 = vD[j + o + 1]
        op = str(OP)
        img = tifffile.TiffFile(g).asarray().transpose([1, 2, 0])
        img2 = tifffile.TiffFile(g2).asarray().transpose([1, 2, 0])
        img3 = tifffile.TiffFile(g3).asarray().transpose([1, 2, 0])
        b = torch.from_numpy((img.astype(float)))
        img3 = torch.from_numpy(img3.astype(float))
        c = torch.from_numpy(img2.astype(float))
        return {'image': b, 'im2': c, 'img3': img3}


# -----------------------------------------------------------------------------
# Load the model
# -----------------------------------------------------------------------------

batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if args.cpu:
    device = 'cpu'
else:
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

U = UNet3D(2, 6)
EX = EXNet(64, 8)
EN = EXNet(64, 6)
U.to(device)
EX.to(device)
EN.to(device)
if device == 'cpu':
    U.load_state_dict(torch.load(mp1,map_location=torch.device('cpu')))
    EX.load_state_dict(torch.load(mp2,map_location=torch.device('cpu')))
    EN.load_state_dict(torch.load(mp3,map_location=torch.device('cpu')))
else:
    U.load_state_dict(torch.load(mp1))
    EX.load_state_dict(torch.load(mp2))
    EN.load_state_dict(torch.load(mp3))
print('    Total params: %.2fM' % (sum(p.numel()
      for p in U.parameters()) / 1000000.0))

val_dataset = VDataset(
    vD, le=1000)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=0)
U.eval()
EN.eval()
EX.eval()

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

def PA(x):
    x = torch.log1p(x)
    return x


tk0 = tqdm(data_loader_val, desc="Iteration")


DK1 = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 2, 1, 1, 0], [1, 1, 2, 3, 2, 1, 1], [
               1, 2, 3, 4, 3, 2, 1], [1, 1, 2, 3, 2, 1, 1], [0, 1, 1, 2, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0]])
DK1 = torch.from_numpy(DK1).float().to(device).reshape(
    [1, 1, 7, 7, 1]) 


def vsup256(U, x, la):
    """
    Run UNet3D on a 256*256*32 patch and extract per-cell detections.

    Parameters
    ----------
    U : nn.Module
        Pretrained 3D U‐Net model.
    x : torch.Tensor, shape (1, 2, 256,256,32)
        2 frames input patch tensor .
    la : torch.Tensor
        Ground‐truth label volume for that patch.

    Returns
    -------
    p1 : Tensor[N,4]
        Coordinates (batch,y,x,z) of detected cells.
    f1 : Tensor[N,C]
        Feature vectors for each detection.
    s1 : Tensor[N]
        Predicted sizes.
    zs1: Tensor[N,2]
        Sigmoid‐normalized probabilities.
    lb1: Tensor[N]
        True labels for each coordinate.
    ar : Tensor[1,H,W,Z]
        Aggregation mask after dilation.
    Uout: Tensor[1,3,H,W,Z]
        Raw U‐Net logits.
    """
    Uout, uo, foz, zs1, size1 = U(x)
    Uf = foz.transpose(0, 1)

    uc = ((search_mask(uo[:, :].max(1)[0]).to(device).max(
        1)[0] == uo[:, 4])) * (Uout.argmax(1) == 1)
    uc[:, :3] = 0
    uc[:, -3:] = 0
    uc[:, :, :3] = 0
    uc[:, :, -3:] = 0
    uc[:, :, :, :3] = 0
    uc[:, :, :, -2:] = 0
    ar = F.conv3d((uc > 0).float().unsqueeze(1), DK1, padding=(3, 3, 0))
    ar = F.conv3d((ar > 0).float(), DK1, padding=(3, 3, 0))
    for i in range(1):
        ar = ud(ar)
    u, x, y, z = torch.where(uc.to(device, dtype=torch.float))
    f1 = Uf[:, u, x, y, z].transpose(0, 1)
    s1 = size1[u, 0, x, y, z]
    zs1 = F.sigmoid(zs1[u, :, x, y, z])
    lb1 = la[u, x, y, z]
    p1 = torch.cat([u.unsqueeze(1), x.unsqueeze(
        1), y.unsqueeze(1), z.unsqueeze(1)], 1)
    return p1, f1, s1, zs1, lb1, ar, Uout


def cp2(x, ba, bb, bc):
    """
    Check whether 2D points lie within given axis-aligned bounds.
    """
    aa = ((x[:, 1]) <= ba.max()) * ((x[:, 1]) >= ba.min())
    ab = ((x[:, 2]) <= bb.max()) * ((x[:, 2]) >= bb.min())
    ac = ((x[:, 3]) <= bc.max()) * ((x[:, 3]) >= bc.min())
    return aa * ab * ac


def cp3(x, ba):
    """
    Check whether 3D points lie inside a binary mask.
    """
    u, x, y, z = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    na = ba[x, y, z] > 0
    return na

# -----------------------------------------------------------------------------
# U-Net feature extraction process for the entire image
# -----------------------------------------------------------------------------


def bvsup256(U, x, l):
    """
    Sliding‐window inference over the full volume via patches.
    
    Parameters
    ----------
    U : nn.Module
        Pretrained 3D U‐Net model.
    x : torch.Tensor, shape (B,2,H,W,Z)
        Full volume input tensor (batch size B).
    l : torch.Tensor
        Full volume ground‐truth labels.
    
    Returns
    -------
    p, f, zs, s, lb, uout : tuple
        Concatenated detection outputs from all patches:
        - p: Tensor[M,4] all coordinates
        - f: Tensor[M,C] all feature vectors
        - zs: Tensor[M,2] all probabilities
        - s: Tensor[M] all sizes
        - lb: Tensor[M] all labels
        - uout: Tensor[H,W,Z] aggregated U‐Net mask
    """
    u, c, pl1, pl2, pl3 = x.shape
    pol = 256
    polz = 32
    pl = pol - 8
    plz = polz - 4
    xm, ym, zm = pl1 // pl + int(pl1 %
                                 pl > 0), pl2 // pl + int(pl2 %
                                                          pl > 0), pl3 // plz + int(pl3 %
                                                                                    plz > 0)
    p, f, s, zs, lb = [], [], [], [], []
    plist = []
    num = 0
    uout = torch.zeros([pl1, pl2, pl3])
    ku = torch.zeros([pl1, pl2, pl3])
    kup = torch.zeros([pl1, pl2, pl3])
    for x1 in range(xm):
        for y1 in range(ym):
            for z1 in range(zm):
                num += 1
                ku[ku > 1] = 1
                v1, v2, v3, v4, v5, v6 = x1 * pl, x1 * pl + pol, y1 * \
                    pl, y1 * pl + pol, z1 * plz, z1 * plz + polz
                if v2 >= pl1:
                    v2 = pl1
                    v1 = v2 - pol
                if v4 >= pl2:
                    v4 = pl2
                    v3 = v4 - pol
                if v6 >= pl3:
                    v6 = pl3
                    v5 = v6 - polz
                with torch.no_grad():
                    p1, f1, s1, zs1, lb1, ar, uo = vsup256(
                        U, x[:, :, v1:v2, v3:v4, v5:v6], l[:, v1:v2, v3:v4, v5:v6])
                uout[v1 + 5:v2 - 5,
                     v3 + 5:v4 - 5,
                     v5:v6] += (uo.argmax(1).squeeze().cpu()[5:-5,
                                                             5:-5] == 1).float()

                ar[ar > 1] = 1
                ar = ar.squeeze().cpu()
                p1[:, 1] += v1
                p1[:, 2] += v3
                p1[:, 3] += v5
                if num > 1:
                    kn = cp3(p1.cpu(), ku == 1)
                    fn = ~kn
                    p1 = p1[fn]
                    f1 = f1[fn]
                    s1 = s1[fn]
                    zs1 = zs1[fn]
                    lb1 = lb1[fn]
                    p = torch.cat([p, p1], 0)
                    f = torch.cat([f, f1], 0)
                    s = torch.cat([s, s1], 0)
                    zs = torch.cat([zs, zs1], 0)
                    lb = torch.cat([lb, lb1], 0)
                else:
                    p = p1
                    f = f1
                    s = s1
                    zs = zs1
                    lb = lb1
                ku[v1:v2, v3:v4, v5:v6] += ar
    return p, f, zs, s, lb, uout

# -----------------------------------------------------------------------------
# Load and test the final frame to determine the processing scope
# -----------------------------------------------------------------------------



img3 = tifffile.TiffFile(vD[269]).asarray().transpose([1, 2, 0]).astype(float)
img4 = tifffile.TiffFile(vD[270]).asarray().transpose([1, 2, 0]).astype(float)
labels = torch.zeros(img3.shape).unsqueeze(0).to(device)
with torch.no_grad():
    p1, f1, zs1, s1, lb1, uo = bvsup256(
        U, (PA(
            torch.cat(
                [
                    torch.from_numpy(img3).to(
                        device, dtype=torch.float).unsqueeze(0).unsqueeze(0), torch.from_numpy(img4).to(
                        device, dtype=torch.float).unsqueeze(0).unsqueeze(0)], 1))), labels)


def box(x):

    x1 = (x > 1)
    l1 = list(x1.sum((0, 1)) > 10)
    z1 = l1.index(1)
    z2 = len(l1) - l1[::-1].index(1)
    l1 = list(x1.sum((1, 2)) > 1)
    xx1 = l1.index(1)
    xx2 = len(l1) - l1[::-1].index(1)
    l1 = list(x1.sum((0, 2)) > 1)
    y1 = l1.index(1)
    y2 = len(l1) - l1[::-1].index(1)

    return xx1, xx2, y1, y2, z1, z2 


x1, x2, y1, y2, z1, z2 = box(uo.cpu())
aq = img3[uo >= 1].min()
win2 = [max(2, x1 - 10), max(2, y1 - 10), 2]
pla = max(256, x2 - x1) + 10
plb = max(256, y2 - y1) + 10
pl = max(38, z2 - z1)

# -----------------------------------------------------------------------------
# Update preprocessing
# -----------------------------------------------------------------------------

def PA(x):
    """
    Preprocess function PA shifts intensities based on data range [2000–65536].  
    - Subtracts min, adds 1900 so minimum maps to 1900.  
    - Applies log1p for compression.  
    These adjustments can better match your training data, though the default 
    (without shift) also often works well.
    """
    x[x < aq] = aq
    x = x - x.min()
    x = x + 1900
    x = torch.log1p(x)
    return x

# -----------------------------------------------------------------------------
# -------------------- Testing Loop --------------------
# This section performs the testing/inference procedure, which includes:
# - Loading input data and processing it patch-wise
# - Using a UNet model to extract features from each patch
# - Performing non-maximum suppression to remove redundant cell centers
#   in both the first and second frames
# - Linking cells between the two frames based on features and positions
# - Aggregating and formatting the results into structured tables
# - During continuous testing, from the second frame onward,
#   reuse the previous second frame's features to avoid redundant
#   re-processing of the first frame

# -----------------------------------------------------------------------------



nid = 0
ZZ = {}
PZ = {}
for step, batch in enumerate(tk0):
# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

    inputs = batch["image"].unsqueeze(1)

    in2 = batch["im2"].unsqueeze(1)

    inputs = inputs[:, :, win2[0]:win2[0] +
                    pla, win2[1]:win2[1] +
                    plb, win2[2]:win2[2] +
                    pl].to(device, dtype=torch.float)

    in2 = in2[:, :, win2[0]:win2[0] + pla, win2[1]:win2[1] +
              plb, win2[2]:win2[2] + pl].to(device, dtype=torch.float)

    in3 = batch["img3"].unsqueeze(1)[
        :,
        :,
        win2[0]:win2[0] +
        pla,
        win2[1]:win2[1] +
        plb,
        win2[2]:win2[2] +
        pl].to(
        device,
        dtype=torch.float)
# -----------------------------------------------------------------------------
# Feature extract
# -----------------------------------------------------------------------------

    with torch.no_grad():
        if step == 0:
            p1, f1, zs1, s1, lb1, _ = bvsup256(
                U, (PA(torch.cat([inputs, in2], 1))), inputs[:, 0] * 0)
        else:
            p1 = zp2
            f1 = zf2
            zs1 = zzsk
            s1 = zzs2
            lb1 = labels[p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 3]]

        p2, f2, zs2, s2, lb2, _ = bvsup256(
            U, (PA(torch.cat([in2, in3], 1))), in2[:, 0] * 0)
    zp2 = p2.clone()
    zf2 = f2.clone()
    zzsk = zs2.clone()
    zzs2 = s2.clone()
    p1 = p1[:, 1:]
    p2 = p2[:, 1:]
    p1[:, 2] = p1[:, 2] * 5
    p2[:, 2] = p2[:, 2] * 5
    qf, ql, px, gf, gl, py = f1, lb1, p1, f2, lb2, p2
    if qf.shape[0] > 2 and gf.shape[0] > 2:
# -----------------------------------------------------------------------------
# Use the MLP to find points belonging to the same cell within the same
# frame and exclude them
# -----------------------------------------------------------------------------
#frame 1
        if step > 0:
            dx = dx2m
            fq = eq.clone()
        else:
            m, n = px.shape[0], px.shape[0]
            px1 = px.clone()
            px1[:, 2] = px1[:, 2] // 5
            distmat = torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(px1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, px1.float(), px1.float().t())
            qx, q = distmat.topk(min(6, px1.shape[0]), largest=False)

            if q.shape[1] < 6:
                while q.shape[1] < 6:
                    q = torch.cat([q, q[:, -1].unsqueeze(-1)], 1)
                    qx = torch.cat([qx, qx[:, -1].unsqueeze(-1)], 1)
            qx = qx[:, 1:]
            q = q[:, 1:]

            ey, epy, esp, ezsp = sort_feature(qf, px, px, q, s1, s1, zs1, zs1, n=5)
            epyy = torch.sqrt((epy * epy).sum(-1))
            with torch.no_grad():
                score = F.sigmoid(
                    EN(qf.unsqueeze(1), ey, epy.float(), esp, ezsp))
            yl = []

            for jk in range(5):
                yl.append((ql[q[:, jk]]).unsqueeze(1))
            yl = torch.cat(yl, 1)
            ss = (score[:, :-1] > 0.5).sum(1)
            ss1 = (score > 0.9999)
            ss2 = (score > 0.6)
            t1 = []
            tx = []
            dx = {}
            dxm = {}
            DD = {}
            for i in range(px.shape[0]):
                if i not in t1:
                    tm = [i]
                    tmm = [ql[i].item()]
                    if ss[i] > 0:
                        ts = ss1[i, :] > ss1[i, -1]
                        ts1 = ss2[i]
                        for j in range(ts.shape[0] - 1):
                            # or (ts1[j] and max(epy[i,j][:2])<2 and
                            # epy[i,j,2]<10)):
                            if ts[j] or (ts1[j] and epyy[i, j]
                                         < max(5, 0.6 * s1[i])):
                                t1.append(q[i, j].item())
                                tm.append(q[i, j].item())
                    dx[i] = tm
                    tx.append(i)
# -----------------------------------------------------------------------------
#frame 2
    
        m, n = py.shape[0], py.shape[0]
        py1 = py.clone()
        py1[:, 2] = py1[:, 2] // 5
        distmat = torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(py1.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, py1.float(), py1.float().t())
        qx, q = distmat.topk(min(6, py1.shape[0]), largest=False)

        if q.shape[1] < 6:
            while q.shape[1] < 6:
                q = torch.cat([q, q[:, -1].unsqueeze(-1)], 1)
                qx = torch.cat([qx, qx[:, -1].unsqueeze(-1)], 1)
        qx = qx[:, 1:]
        q = q[:, 1:]
        eq = q.clone()

        ey,epy,esp,ezsp = sort_feature(gf, py, py, q, s2, s2, zs2, zs2, n=5)
        epyy = torch.sqrt((epy * epy).sum(-1))
        with torch.no_grad():
            score = F.sigmoid(EN(gf.unsqueeze(1), ey, epy.float(), esp, ezsp))
        yl = []

        for jk in range(5):
            yl.append((gl[q[:, jk]]).unsqueeze(1))
        yl = torch.cat(yl, 1)
        ss = (score[:, :-1] > 0.5).sum(1)
        ss1 = (score > 0.9999)
        ss2 = (score > 0.6)
        t1 = []
        tx = []
        dx2 = {}
        dx2m = {}
        DD2 = {}
        dxf = {}
        for i in range(py.shape[0]):
            # if i not in t1:
            tm = [i]
            tmm = [i]
            dxf[i] = i
            if ss[i] > 0:
                ts = ss1[i, :] > ss1[i, -1]
                ts1 = ss2[i]
                for j in range(ts.shape[0] - 1):
                    # or (ts1[j] and max(epy[i,j][:2])<2 and epy[i,j,2]<10)):
                    if ts[j] or (ts1[j] and epyy[i, j] < max(5, 0.6 * s2[i])):

                        tm.append(q[i, j].item())
                        dxf[q[i, j].item()] = i
                        if i not in t1:
                            tmm.append(q[i, j].item())
                        t1.append(q[i, j].item())
            dx2[i] = tm
            if i not in t1:
                dx2m[i] = tmm
            fm = {}
            for i in dx2m:
                for j in dx2m[i]:
                    fm[j] = i
            tx.append(i)

# -----------------------------------------------------------------------------
# Link cells across different frames
# -----------------------------------------------------------------------------
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, px.float(), py.float().t())
        qx, q = distmat.topk(min(5, px.shape[0], py.shape[0]), largest=False)

        if q.shape[1] < 5:
            while q.shape[1] < 5:
                q = torch.cat([q, q[:, -1].unsqueeze(-1)], 1)
                qx = torch.cat([qx, qx[:, -1].unsqueeze(-1)], 1)


        ey, epy, esp, ezsp = sort_feature(gf, py, px, q, s2, s1, zs2, zs1, n=5)
        epyy = torch.sqrt((epy * epy).sum(-1))
        with torch.no_grad():
            score = (EX(qf.unsqueeze(1), ey, epy.float(), esp, ezsp))
            sp = score[:, -2:]
            score = F.sigmoid(score[:, :-2])
        yl = []
        for jk in range(5):
            yl.append((gl[q[:, jk]]).unsqueeze(1))
        yl = torch.cat(yl, 1)
        t1 = []
        tx = []
        fp = 0
        fn = 0
        fpd = 0
        fnd = 0
        IS = 0
        c = F.softmax(sp)
        cc = 0
        ss1 = (score > 0.5)
        r = {}
        r1s = {}
        r2 = {}
        r2s = {}
        r1l = {}
        r2l = {}
        r1f = {}
        r2f = {}
        rk = {}
        r1su = {}
        plq = {}
        nn = 0
        scorei = score
        for di in dx:
            nn += 1
            lq = min(dx[di])
            if step > 0:
                lq = qfm[min(dx[di])]
            if 1:
                rt = []
                rs = []
                res = []
                g = []
                su = np.max([s1[i].item() for i in dx[di]])

                plq[lq] = torch.cat([p1[i].unsqueeze(0)
                                    for i in dx[di]], 0).float().mean(0).cpu().numpy()

                for i in dx[di]:
                    ts = ss1[i, :]
                    g.append(c[i][1].item())
                    if score[i].argmax(0) < 5:
                        ss = scorei[i].topk(5)[1]
                        zz = dx2[q[i, ss[0]].item()]
                        rt.append(q[i, ss[0]].item())
                        rs.append(scorei[i, ss[0]].item())
                        res.append(epyy[i, ss[0]].item())
                        for st in ss[1:]:
                            if st == 5:
                                break
                            if q[i, st].item() not in zz:
                                rt.append(q[i, st].item())
                                rs.append(scorei[i, st].item())
                                res.append(epyy[i, st].item())
                                zz = zz + dx2[q[i, st].item()]

                if len(rs) > 0:
                    rk = rs.index(np.max(rs))
                    dj = {}
                    djn = {}
                    for ie in range(len(rt)):
                        dj[rt[ie]] = max(dj.get(rt[ie], 0), rs[ie])
                        djn[rt[ie]] = res[ie]
                    nrt = list(dj.keys())
                    nrs = [dj[ie] for ie in nrt]
                    nres = [djn[ie] for ie in nrt]
                    if 1:
                        dx2[100000] = []
                        rtx = []
                        rts = []
                        zr = []
                        zrs = []
                        zrl = []
                        rtl = []
                        zr2 = r.get(lq, [100000, 100000])
                        zr2s = r1s.get(lq, [0, 0])
                        zr2l = r1l.get(lq, [0, 0])
                        zr2.append(100000)
                        zr2s.append(0)
                        zr2l.append(0)
                        nrt = nrt + zr2
                        nrs = nrs + zr2s
                        nres = nres + zr2l
                        ss = torch.from_numpy(np.array(nrs)).topk(
                            min(5, len(nrs)))[1]
                        zz = []

                        rtx.append(nrt[ss[0]])
                        rts.append(nrs[ss[0]])
                        rtl.append(nres[ss[0]])

                        zz = zz + dx2[nrt[ss[0]]]
                        for st in ss[1:]:
                            if nrt[st] == 100000:
                                break
                            if nrt[st] not in zz and nres[st] < max(su, 5) * 6:
                                rtx.append(nrt[st])
                                rts.append(nrs[st])
                                rtl.append(nres[st])
                                zz = zz + dx2[nrt[st]]
                        r[lq] = rtx[:2]
                        r1s[lq] = rts[:2]
                        r1l[lq] = rtl[:2]
                        if r1f.get(lq, 0) < np.max(g):
                            r1f[lq] = np.max(g)
                        r1su[lq] = su

        el = []
        for i in r:
            el.append(r[i])
            r[i] = [[fm.get(r[i][x], -1), r1s[i][x]] for x in range(min(len(r[i]), 1 + int(r1f[i] > 0.8 or (r1s[i][-1] > 0.5 and r1f[i]
                                                                                                            > 0.5 and np.abs(r1l[i][-1] - r1l[i][0]) < 1.2 * r1su[i])))) if x != 100000 and ((x == 0 and r1s[i][0] > 0.1) or (x > 0))]
        zr = {}

        for i in r:
            for j in r[i]:
                if zr.get(j[0], 0) < j[1]:
                    zr[j[0]] = j[1]

        for i in r:
            for j in r[i]:
                if zr.get(j[0], 0) > j[1]:
                    r[i].remove(j)
        r0 = r.copy()
# -----------------------------------------------------------------------------
# Reconnect broken trajectories
# -----------------------------------------------------------------------------
        if step > 0:
            qfmv = list(qfm.values())
            for i in r:
                if len(r[i]) > 0:
                    if i not in qr:
                        if (fq[i][0].item() in qfmv):
                            if qfm[fq[i][0].item()] in qr and qfm[fq[i]
                                                                  [0].item()] in plq:
                                tf = fq[i][0].item()
                                r0[tf] = r0.get(tf, []) + r[i]
                                r0.pop(i)

                        elif fq[i][1].item() in qfmv:
                            if qfm[fq[i][1].item()] in qr and qfm[fq[i]
                                                                  [1].item()] in plq:
                                tf = fq[i][1].item()
                                r0[tf] = r0.get(tf, []) + r[i]
                                r0.pop(i)

                        elif fq[i][2].item() in qfmv:
                            if qfm[fq[i][2].item()] in qr and qfm[fq[i]
                                                                  [2].item()] in plq:
                                tf = fq[i][2].item()
                                r0[tf] = r0.get(tf, []) + r[i]
                                r0.pop(i)
# -----------------------------------------------------------------------------
# Log data
# -----------------------------------------------------------------------------
        rr = {}
        pr = {}
        qr = []
        for i in r0:
            if i in plq:
                i0 = i
                rr[i0] = [j[0] for j in r0[i]]
                qr += [j[0] for j in r0[i]]
                pr[i0] = plq[i0] + win2
        ZZ[step] = rr
        PZ[step] = pr
        qfm = fm.copy()
# -----------------------------------------------------------------------------
# Integrate data, remove excessively short paths, and output trajectories
# -----------------------------------------------------------------------------
tit = {}
cidn = 1
tidn = 1
cid = {}
tid = {}
t = 0
qid = {}
pid = {}
did = {}
temp = PZ[t].copy()
tmp = {}
for i in ZZ[t]:
    if (isinstance(ZZ[t][i], list) and len(ZZ[t][i]) > 0):
        cid[cidn] = [cidn, tidn, t, temp[i][0].item(), temp[i][1].item(),
                     temp[i][2].item(), -1]
        tid[tidn] = tid.get(tidn, []) + [cidn]
        tit[tidn] = tit.get(tidn, []) + [t]
        for j in ZZ[t][i].copy():
            tmp[j] = cidn
        cidn += 1
        tidn += 1


t = 1
for t in tqdm(range(1, 267)):
    ntmp = {}
    for i in ZZ[t]:

        temp = PZ[t]
        bemp = PZ[t - 1]
        pi = tmp.get(i, -1)
        if pi in cid:
            ti = cid[pi][1]
        else:
            tidn += 1
            ti = tidn
        cid[cidn] = [
            cidn,
            ti,
            t,
            temp[i][0].item(),
            temp[i][1].item(),
            temp[i][2].item(),
            pi]
        tid[ti] = tid.get(ti, []) + [cidn]
        tit[ti] = tit.get(ti, []) + [t]
        for j in ZZ[t][i].copy():
            ntmp[j] = cidn
        cidn += 1
    tmp = ntmp.copy()

plink = {}
for i in cid:
    plink[i] = cid[i][-1]
link = {}
for i in plink:
    link[plink[i]] = link.get(plink[i], []) + [i]
no = []
for i in tit:
    if len(
        tit[i]) < 3 and max(
        tit[i]) < 260 and max(
            tid[i]) not in link and min(
                tit[i]) > 0:
        no.append(i)
for i in cid.copy():
    if cid[i][1] in no:
        cid.pop(i)

D = pd.DataFrame()
lt, l1, l2, l3, l4, l5, l6, l7 = [], [], [], [], [], [], [], []
for i in cid:
    pi = cid[i]

    l1.append(pi[0])
    l2.append(pi[1])
    l3.append(pi[2])
    l4.append(pi[3])
    l5.append(pi[4])
    l6.append(pi[5])
    l7.append(pi[6])

D['cellid'] = l1
D['trackid'] = l2
D['t'] = l3
D['x'] = l4
D['y'] = l5
D['z'] = l6
D['parentid'] = l7

D.to_csv('track-result.csv', index=False)
