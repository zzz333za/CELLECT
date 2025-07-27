
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
parser.add_argument('--train', type=str, required=True, help="Train data id")

parser.add_argument('--val', type=str, required=True, help="Val data id")
parser.add_argument('--model_dir', type=str, required=True,
                    help="Path to the output models directory")

args = parser.parse_args()
datap = args.data_dir
outp = args.out_dir
mp = args.model_dir
op1 = int(args.train)

op2 = int(args.val)
if not os.path.exists(mp):
    os.mkdir(mp)

# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
op = str(op1)
l = os.listdir(datap + '/mskcc_confocal_s' + op + '/images/')
l = [i for i in l if 'tif' in i]

D = {}
om = os.listdir(outp + '/ls' + op + '/')
for i in tqdm(l):
    if 'tif' in i:

        num = int(i.split('_')[-1].split('.')[0][1:])
        if num < 275:

            D[num] = datap + '/mskcc_confocal_s' + op + '/images/' + i


d1 = pd.read_table(datap + '/mskcc_confocal_s' + op + '/tracks/tracks.txt')
d2 = pd.read_table(datap + '/mskcc_confocal_s' + op +
                   '/tracks/tracks_polar_bodies.txt')

op = str(op2)
vl = os.listdir(datap + '/mskcc_confocal_s' + op + '/images/')
vl = [i for i in vl if 'tif' in i]
om = os.listdir(outp + '/ls' + op + '/')
vD = {}
for i in tqdm(vl):
    if 'tif' in i:

        num = int(i.split('_')[-1].split('.')[0][1:])
        if num < 275:

            vD[num] = datap + '/mskcc_confocal_s' + op + '/images/' + i


vd1 = pd.read_table(datap + '/mskcc_confocal_s' + op + '/tracks/tracks.txt')
vd2 = pd.read_table(datap + '/mskcc_confocal_s' + op +
                    '/tracks/tracks_polar_bodies.txt')

ii1 = int(vd1.y.min() - 30)
ii2 = max(int(vd1.y.max() + 30), ii1 + 256)
ii3 = int(vd1.x.min() - 30)
ii4 = max(ii3 + 256, int(vd1.x.max() + 30))
ii5 = 5
win2 = [min(max(0, (ii2 - ii1 - 256) // 2), 30),
        min(max(0, (ii4 - ii3 - 256) // 2), 30), 2]
# -----------------------------------------------------------------------------
# Configure the data loader
# -----------------------------------------------------------------------------
class IntracranialDataset(Dataset):

    def __init__(self, data, le):

        self.data = data

    def __len__(self):

        return len(self.data) - 3 + 100

    def __getitem__(self, i):

        if i < 268:

            j = i
        else:
            j = random.choice(list(range(250, 272, 1)))
        g = D[j]
        o = 1
        g2 = D[j + o]
        g3 = D[j + o + 1]
        op = str(op1)

        img = tifffile.TiffFile(g).asarray().transpose([1, 2, 0])
        img2 = tifffile.TiffFile(g2).asarray().transpose([1, 2, 0])
        img3 = tifffile.TiffFile(g3).asarray().transpose([1, 2, 0])
        t = np.load(outp + '/ls' + op + '/' + str(j) + '-k1-3d-1-imaris.npy')
        t1 = np.load(outp + '/ls' + op + '/' + str(j) + '-k2-3d-1-imaris.npy')
        t4 = np.load(outp + '/ls' + op + '/' + str(j) + '-k5-3d-1-imaris.npy')
        t5 = np.load(outp + '/ls' + op + '/' + str(j) + '-k6-3d-1-imaris.npy')
        t6 = np.load(outp + '/ls' + op + '/' + str(j) + '-k1-3d-imaris.npy')
        t7 = np.load(outp + '/ls' + op + '/' + str(j) + '-k2-3d-imaris.npy')
        t8 = np.load(outp + '/ls' + op + '/' + str(j) + '-k3-3d-imaris.npy')
        t9 = np.load(outp + '/ls' + op + '/' + str(j) + '-k4-3d-imaris.npy')
        t15 = np.load(outp + '/ls' + op + '/' +
                      str(j + 1) + '-k6-3d-1-imaris.npy')
        t31 = np.load(outp + '/ls' + op + '/' + str(j) + '-k31-imaris.npy')
        t32 = np.load(outp + '/ls' + op + '/' + str(j) + '-k32-imaris.npy')
        xa, xb, xc = random.randint(50, max(0, t.shape[0] - 256)), random.randint(
            50, max(0, t.shape[1] - 256)), random.randint(3, max(0, t.shape[2] - 32))
        pa, pb, pc = 256, 256, 32
        b = torch.from_numpy((img.astype(float)))
        timg = torch.from_numpy(t)
        img3 = torch.from_numpy(img3.astype(float))
        c = torch.from_numpy(img2.astype(float))
        d = torch.from_numpy(t1)
        e1 = torch.from_numpy(t4)
        e2 = torch.from_numpy(t5)
        e4 = torch.from_numpy(t15)
        p1 = torch.from_numpy(t6)
        p2 = torch.from_numpy(t7)
        p3 = torch.from_numpy(t8)
        p4 = torch.from_numpy(t9)
        size1 = torch.from_numpy(t31)
        size2 = torch.from_numpy(t32)
        return {'image': b[xa:xa + pa,
                           xb:xb + pb,
                           xc:xc + pc],
                'labels': p1[xa:xa + pa,
                             xb:xb + pb,
                             xc:xc + pc],
                'im2': c[xa:xa + pa,
                         xb:xb + pb,
                         xc:xc + pc],
                'la2': p2[xa:xa + pa,
                          xb:xb + pb,
                          xc:xc + pc],
                'p1': timg[xa:xa + pa,
                           xb:xb + pb,
                           xc:xc + pc],
                'p2': d[xa:xa + pa,
                        xb:xb + pb,
                        xc:xc + pc],
                's1': e1[xa:xa + pa,
                         xb:xb + pb,
                         xc:xc + pc],
                's2': e2[xa:xa + pa,
                         xb:xb + pb,
                         xc:xc + pc],
                's4': e4[xa:xa + pa,
                         xb:xb + pb,
                         xc:xc + pc],
                'la3': p3[xa:xa + pa,
                          xb:xb + pb,
                          xc:xc + pc],
                'la4': p4[xa:xa + pa,
                          xb:xb + pb,
                          xc:xc + pc],
                'size1': size1[xa:xa + pa,
                               xb:xb + pb,
                               xc:xc + pc],
                'size2': size2[xa:xa + pa,
                               xb:xb + pb,
                               xc:xc + pc],
                'img3': img3[xa:xa + pa,
                             xb:xb + pb,
                             xc:xc + pc]}


class VDataset(Dataset):

    def __init__(self, data, le):

        self.data = data

    def __len__(self):

        return len(self.data) - 203

    def __getitem__(self, i):

        j = i + 200
        g = vD[j]
        o = 1
        g2 = vD[j + o]
        g3 = vD[j + o + 1]
        op = str(op2)
        img = tifffile.TiffFile(g).asarray().transpose([1, 2, 0])
        img2 = tifffile.TiffFile(g2).asarray().transpose([1, 2, 0])
        img3 = tifffile.TiffFile(g3).asarray().transpose([1, 2, 0])
        t = np.load(outp + '/ls' + op + '/' + str(j) + '-k1-3d-1-imaris.npy')
        t1 = np.load(outp + '/ls' + op + '/' + str(j) + '-k2-3d-1-imaris.npy')
        t4 = np.load(outp + '/ls' + op + '/' + str(j) + '-k5-3d-1-imaris.npy')
        t5 = np.load(outp + '/ls' + op + '/' + str(j) + '-k6-3d-1-imaris.npy')
        t6 = np.load(outp + '/ls' + op + '/' + str(j) + '-k1-3d-imaris.npy')
        t7 = np.load(outp + '/ls' + op + '/' + str(j) + '-k2-3d-imaris.npy')
        t15 = np.load(outp + '/ls' + op + '/' +
                      str(j + 1) + '-k6-3d-1-imaris.npy')
        t31 = np.load(outp + '/ls' + op + '/' + str(j) + '-k31-imaris.npy')
        t32 = np.load(outp + '/ls' + op + '/' + str(j) + '-k32-imaris.npy')
        b = torch.from_numpy((img.astype(float)))
        timg = torch.from_numpy(t)
        img3 = torch.from_numpy(img3.astype(float))
        c = torch.from_numpy(img2.astype(float))
        d = torch.from_numpy(t1)

        e1 = torch.from_numpy(t4)
        e2 = torch.from_numpy(t5)

        e4 = torch.from_numpy(t15)
        p1 = torch.from_numpy(t6)
        p2 = torch.from_numpy(t7)
        size1 = torch.from_numpy(t31)
        size2 = torch.from_numpy(t32)
        return {
            'image': b,
            'labels': p1,
            'im2': c,
            'la2': p2,
            'p1': timg,
            'p2': d,
            's1': e1,
            's2': e2,
            'img3': img3,
            's4': e4,
            'size1': size1,
            'size2': size2}



# -----------------------------------------------------------------------------
# Set training environment parameters
# -----------------------------------------------------------------------------
SMOOTH = 1e-6
n_epochs = 150
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ids = [0]
device = torch.device("cuda:0")
device1 = torch.device("cpu")
tm = 2
tm1 = 1
DK = np.zeros([tm * 2 + 1, tm * 2 + 1, 9])
DK = torch.from_numpy(DK).float().to(device).reshape(
    [1, 1, tm * 2 + 1, tm * 2 + 1, 9])
DK1 = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 2, 1, 0, 0], [0, 1, 2, 3, 2, 1, 0], [
               1, 2, 3, 4, 3, 2, 1], [0, 1, 2, 3, 2, 1, 0], [0, 0, 1, 2, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
DK1 = torch.from_numpy(DK1).float().to(device).reshape([1, 1, 7, 7, 1])

U = UNet3D(2, 6)
EX = EXNet(64, 8)
EN = EXNet(64, 6)

U.to(device)
EX.to(device)
EN.to(device)

plist = [{'params': U.parameters()}]
oU = optim.Adam(plist, lr=2e-4)
plist = [{'params': EX.parameters(), 'lr': 2e-4}]
oEX = optim.Adam(plist)
plist = [{'params': EN.parameters(), 'lr': 2e-4}]
oEN = optim.Adam(plist)

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

sU = torch.optim.lr_scheduler.ReduceLROnPlateau(oU, factor=0.5, patience=3)
sEX = torch.optim.lr_scheduler.ReduceLROnPlateau(oEX, factor=0.5, patience=3)
sEN = torch.optim.lr_scheduler.ReduceLROnPlateau(oEN, factor=0.5, patience=3)


print('    Total params: %.2fM' % (sum(p.numel()
      for p in U.parameters()) / 1000000.0))

l0 = []
l1 = []
train_dataset = IntracranialDataset(
    D, le=1000)
data_loader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = VDataset(
    vD, le=1000)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=0)
val_loss = 1000
va = 10000
win = 3
lg = 16
ko = 3
pk = (1, 1, 1)



# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
def PA(x):
    x = torch.log1p(x)
    return x

# -----------------------------------------------------------------------------
# Trainning
# -----------------------------------------------------------------------------
for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    U.train()
    if epoch < 6 or (epoch > 15 and epoch <= 20):
        EX.train()
        EN.train()

    else:

        EX.eval()
        EN.eval()
    tr_loss = 0
    kk = 0

    tk0 = tqdm(data_loader_train, desc="Iteration")
    n = 0
# -----------------------------------------------------------------------------
# -------------------- Training Loop --------------------
# This section performs the training procedure, which includes:
# - Loading input data batches from the dataloader
# - Using a UNet model to extract per-frame features
# - Computing full-image predictions: segmentation map, confidence map,
#   cell size, and mitosis (division) indicators
# - Calculating corresponding losses for these outputs and optimizing them
# - Identifying individual cell centers from the feature maps
# - For each cell, computing classification, matching, and contrastive losses
#   using MLP and contrastive embedding learning
# - Backpropagation and optimizer steps to update the model
# -----------------------------------------------------------------------------
    for step, batch in enumerate(tk0):
# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
        if step % 10 == 0:
            print(tk0)
        inputs = batch["image"].unsqueeze(1)

        labels = batch["labels"]
        in2 = batch["im2"].unsqueeze(1)
        la2 = batch["la2"]
        la3 = batch["la3"]
        la4 = batch["la4"]
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        in2 = in2.to(device, dtype=torch.float)
        la2 = la2.to(device, dtype=torch.float)
        la3 = la3.to(device, dtype=torch.float)
        la4 = la4.to(device, dtype=torch.float)
        p1 = batch["p1"].to(device, dtype=torch.float).unsqueeze(1)
        p2 = batch["p2"].to(device, dtype=torch.float).unsqueeze(1)
        pl1 = batch["p1"].to(device, dtype=torch.float).unsqueeze(1)
        pl2 = batch["p2"].to(device, dtype=torch.float).unsqueeze(1)
        ee1 = list((pl1[0, 0][pl1[0, 0] != 0]).flatten().cpu().numpy())
        ee2 = list((pl2[0, 0][pl2[0, 0] != 0]).flatten().cpu().numpy())
        lx2 = torch.unique(pl1[0, 0]).shape[0]
        w3 = (lx2 > 400) * 1

        lsize1 = batch["size1"].to(device, dtype=torch.float)
        lsize2 = batch["size2"].to(device, dtype=torch.float)
        s1 = batch["s1"].to(device, dtype=torch.float)
        s2 = batch["s2"].to(device, dtype=torch.float)

        s4 = batch["s4"].to(device, dtype=torch.float)
        in3 = batch["img3"].unsqueeze(1).to(device, dtype=torch.float)
# -----------------------------------------------------------------------------
# Run the main model to obtain outputs for:
# - Cell segmentation
# - Center point scoring
# - Division likelihood assessment
# - Size estimation
# -----------------------------------------------------------------------------
        Uout, uo, fo, zs1, size1 = U(PA(torch.cat([inputs, in2], 1)))
        Uout2, uo2, fo2, zs2, size2 = U(PA(torch.cat([in2, in3], 1)))
        p1[:, 0] = F.conv3d((p1[:, 0] > 0).float().unsqueeze(
            1), DK1, padding=(tm + 1, tm + 1, 0))
        p2[:, 0] = F.conv3d((p2[:, 0] > 0).float().unsqueeze(
            1), DK1, padding=(tm + 1, tm + 1, 0))
        for i in range(3):
            p1 = ud(p1)
            p2 = ud(p2)
        zzs = s1.clone()
        s1 = torch.cat([(s1 == 1).float(), (s2 == 2)], 0)

        s2 = torch.cat([(s2 == 1).float(), (s4 == 2)], 0)

        la3 = (labels > 0).float() + 2 * (la3 > 0).float()
        la3[la3 > 2] = 2
        la4 = (la2 > 0).float() + 2 * (la4 > 0).float()
        la4[la4 > 2] = 2
        wp1 = la3 > 0
        zp1 = wp1 * (p1[:, 0])
        wp2 = la4 > 0
        zp2 = wp2 * (p2[:, 0])
# -----------------------------------------------------------------------------
# Segmentation loss
# -----------------------------------------------------------------------------
        nclc = criterion3(Uout, (la3).long())
        nclc2 = criterion3(Uout2, (la4).long())
        w1 = labels.clone()
        w2 = la2.clone()

        ws = w1 > 0
        ws2 = w2 > 0
# -----------------------------------------------------------------------------
# Size loss
# -----------------------------------------------------------------------------
        lclc = mse(size1, lsize1)
        lclc2 = mse(size2, lsize2)

        sloss = torch.mean(lclc) + torch.mean(lclc2)
# -----------------------------------------------------------------------------
# Division loss
# -----------------------------------------------------------------------------
        sloss1 = torch.mean(
            (bce(F.sigmoid(zs1), s1.unsqueeze(0)) * (1 + s1 * 15)))
        sloss1 = sloss1 * (s1.sum() > 0)
        rloss3 = 0
        kloss = []
# -----------------------------------------------------------------------------
# Loss computation based on given cell points
# Extract information for each individual cell
# -----------------------------------------------------------------------------
        Uf = fo.transpose(0, 1)
        Uf2 = fo2.transpose(0, 1)
# -----------------------------------------------------------------------------
# Center point extraction
# -----------------------------------------------------------------------------
        uc = ((search_mask(uo[:, :].max(1)[0]).cuda().max(1)[0] ==
              uo[:, 4])) * (Uout.argmax(1) == 1) 
        u, x, y, z = torch.where(uc.to(device, dtype=torch.float))
        lx = torch.unique(labels).shape[0]
        lx1 = torch.unique(labels * uc).shape[0]
        lx3 = list(torch.unique(labels * uc).cpu().numpy())
        lx3 = [i for i in lx3 if i in np.unique(ee1)]

        f1 = Uf[:, u, x, y, z].transpose(0, 1)
        s1 = size1[u, 0, x, y, z]
        zs1 = F.sigmoid(zs1)[u, :, x, y, z]

        for i in ee1:
            if ee2.count(i) > 1 or i not in lx3:
                w1[w1 == i] = 4
                w2[w2 == i] = 4

        w1[w1 != 4] = 1
        w2[w2 != 4] = 1

        lb1 = labels[u, x, y, z]
        zb = zzs[u, x, y, z] == 1
        uc = ((search_mask(uo2[:, :].max(1)[0]).cuda().max(1)[0] ==
              uo2[:, 4])) * (Uout2.argmax(1) == 1)  

        nloss1 = crloss(uo, (zp1).long()) * w1
        nloss2 = crloss((uo2), (zp2).long()) * w2
        nloss3 = torch.mean(nloss1[labels > 0])
        nloss4 = torch.mean(nloss2[la2 > 0])
        Uloss = torch.mean(nloss1) + torch.mean(nloss2) + nloss3 + \
            nloss4 + torch.mean(nclc * w1) + torch.mean(nclc2 * w2)

        Uloss = Uloss * (1 + w3)
# -----------------------------------------------------------------------------
# All subsequent losses are computed on a per-cell basis (cell-wise).
# -----------------------------------------------------------------------------
        if u.shape[0] > 0 and epoch > 1:

            u2, x2, y2, z2 = torch.where(uc.to(device, dtype=torch.float))
            f2 = Uf2[:, u2, x2, y2, z2].transpose(0, 1)
            s2 = size2[u2, 0, x2, y2, z2]
            zs2 = F.sigmoid(zs2)[u2, :, x2, y2, z2]
            lb2 = la2[u2, x2, y2, z2]

            p1 = torch.cat([u.unsqueeze(1), x.unsqueeze(
                1), y.unsqueeze(1), z.unsqueeze(1)], 1)
            p2 = torch.cat([u2.unsqueeze(1), x2.unsqueeze(
                1), y2.unsqueeze(1), z2.unsqueeze(1)], 1)

            p1 = p1[:, 1:]
            p2 = p2[:, 1:]
            p1[:, 2] = p1[:, 2] * 5# resolution z/xy
            p2[:, 2] = p2[:, 2] * 5
            px = p1.float()
            py = p2.float()
# -----------------------------------------------------------------------------
# Set an error tolerance range to match the extracted center points
# with the sparsely annotated center points
# -----------------------------------------------------------------------------
            qf, ql, px, gf, gl, py = f1, lb1, p1, f2, lb2, p2
            if qf.shape[0] > 5 and gf.shape[0] > 5:
# -----------------------------------------------------------------------------
# frame 1

                ql = sort_labels(px, pl1, ql, ee1, ratio=5)


# -----------------------------------------------------------------------------
# frame 2
                    
                gl = sort_labels(py, pl2, gl, ee2, ratio=5)



# -----------------------------------------------------------------------------
# Train the MLP model to distinguish identical cells within the
# same frame
# -----------------------------------------------------------------------------
                m, n = px.shape[0], px.shape[0]
                distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px.float(), px.float().t())
                qx, q = distmat.topk(6, largest=False)
                qx = qx[:, 1:]
                q = q[:, 1:]
                px = px.to(device)
                py = py.to(device)

                sx = [0, 1, 2, 3, 4]
                if random.randint(0, 1) == 1:
                    nq = q.clone()
                    nqx = qx.clone()
                    for ki in range(q.shape[0]):
                        random.shuffle(sx)
                        for jk in range(5):
                            nq[ki, jk] = q[ki, sx[jk]]

                            nqx[ki, jk] = qx[ki, sx[jk]]

                    q = nq
                    qx = nqx

                ey, epy, esp, ezsp = sort_feature(qf, px, px, q, s1, s1, zs1, zs1, n=5)
                score = EN(qf.unsqueeze(1), ey, epy.float(), esp, ezsp)
                yl = []
                ya = 0
                for jk in range(5):
                    yl.append((ql[q[:, jk]] == ql).unsqueeze(1))
                    ya += (ql[q[:, jk]] == ql).float()

                yl.append((ya == 0).unsqueeze(1))
                yl = torch.cat(yl, 1)

                score = score[ql != 0]
                yl = yl[ql != 0]

                tkloss = torch.mean(bce(F.sigmoid(score), yl.float()))


# -----------------------------------------------------------------------------
# Train the MLP model to distinguish identical cells across different frames
# and identify divided cells.
# The difference compared to another model lies in whether divided cells
# are considered the same cell.
# -----------------------------------------------------------------------------
                px = px.to(device1)
                py = py.to(device1)
                ws = np.array([ee2.count(i) for i in ql])
                ws = (ws > 1) * 9 + 1
                m, n = px.shape[0], py.shape[0]
                distmat = torch.pow(px.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(py.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, px.float(), py.float().t())
                qx, q = distmat.topk(5, largest=False)

                px = px.to(device)
                py = py.to(device)

                sx = [0, 1, 2, 3, 4]
                if random.randint(0, 1) == 1:
                    nq = q.clone()
                    nqx = qx.clone()
                    for ki in range(q.shape[0]):
                        random.shuffle(sx)
                        for jk in range(5):
                            nq[ki, jk] = q[ki, sx[jk]]

                            nqx[ki, jk] = qx[ki, sx[jk]]

                    q = nq
                    qx = nqx
                ey, epy, esp, ezsp = sort_feature(gf, py, px, q, s2, s1, zs2, zs1, n=5)
                score = EX(qf.unsqueeze(1), ey, epy.float(), esp, ezsp)
                sp = score[:, -2:]
                score = score[:, :-2]
                yl = []
                ya = 0
                for jk in range(5):
                    yl.append((gl[q[:, jk]] == ql).unsqueeze(1))
                    ya += (gl[q[:, jk]] == ql).float()
                zcs = criterion6(sp, zb.long())
                yl.append((ya == 0).unsqueeze(1))
                yl = torch.cat(yl, 1)
                yls = yl[:, :5].float().sum(1)
                ylmax = yl.float().argmax(1)
                ylmin = yl[:, :5].float().argmin(1)
                score = score[ql != 0]
                yls = yls[ql != 0]
                yl = yl[ql != 0]
                ws = ws[ql.cpu() != 0]
                cl = bce(F.sigmoid(score), yl.float())
                cl[ws > 1] = cl[ws > 1] * 4

                tkloss += torch.mean(cl) * 5 + 5 * torch.mean(zcs)
                zz = sp.argmax(1)

                tcloss = 0

                if tkloss.item() == tkloss.item():

                    kloss.append((tcloss + tkloss).unsqueeze(-1))

# -----------------------------------------------------------------------------
# Contrastive learning phase
# Obtain equal amounts of data pairs for each class across two frames:
# - For the same cell, one sample per frame (total of 2 samples).
# - For divided cells, three samples per instance.
# -----------------------------------------------------------------------------
            lb1 = ql
            lb2 = gl
            til = list((lb1).cpu().numpy())
            z1 = pd.value_counts(til).to_dict()
            til = list((lb2).cpu().numpy())

            z2 = pd.value_counts(til).to_dict()
            cb1 = lb1.cpu().numpy().astype(int)
            cb2 = lb2.cpu().numpy().astype(int)

            tt = []
            lt = []
            for ix in range(len(lb1)):
                if cb1[ix] != 0 and z1.get(
                        cb1[ix], 0) >= 1 and z2.get(
                        cb1[ix], 0) >= 2 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])
            tt = np.array(tt)
            f1m = f1[tt]
            p1m = p1[tt]
            lb1m = lb1[tt]

            tt = []
            lt = []
            for ix in range(len(lb1)):
                if cb1[ix] != 0 and z1.get( cb1[ix], 0) >= 1 and z2.get(
                        cb1[ix], 0) >= 1 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])

            tt = np.array(tt)
            f1 = f1[tt]
            p1 = p1[tt]
            lb1 = lb1[tt]
            cb1 = lb2.cpu().numpy()

            tt = []
            lt = []
            tt1 = []
            for ix in range(len(lb2)):
                if cb1[ix] != 0 and z1.get( cb1[ix], 0) >= 1 and z2.get( cb1[ix], 0) >= 2:
                    if cb1[ix] not in lt:
                        tt.append(ix)
                        lt.append(cb1[ix])
                    elif lt.count(cb1[ix]) < 2:
                        tt1.append(ix)
                        lt.append(cb1[ix])
            tt = np.array(tt)
            tt1 = np.array(tt1)
            f2m = f2[tt]
            p2m = p2[tt]
            lb2m = lb2[tt1]
            f2m2 = f2[tt1]
            p2m2 = p2[tt1]
            lb2m2 = lb2[tt1]
            tt = []
            lt = []
            for ix in range(len(lb2)):
                if cb1[ix] != 0 and z1.get(
                        cb1[ix], 0) >= 1 and z2.get(
                        cb1[ix], 0) >= 1 and cb1[ix] not in lt:
                    tt.append(ix)
                    lt.append(cb1[ix])
            tt = np.array(tt)
            f2 = f2[tt]
            p2 = p2[tt]
            lb2 = lb2[tt]

            ZL = [[[f1, f2], [lb1, lb2]], [[f2m, f2m2], [lb2m, lb2m2]], [[f1m, torch.cat(
                [f2m, f2m2], 0)], [lb1m, torch.cat([lb2m, lb2m2], 0)]]]  # ,[[bf,bf2],[blb,blb2]]]
# -----------------------------------------------------------------------------
# Compute triplet loss
# -----------------------------------------------------------------------------
            for inx in ZL:

                Ff, L = inx
                Ff = torch.cat(Ff, 0)
                L1 = torch.cat(L, 0)
                if Ff.shape[0] > 4:
                    lt = Tloss(Ff, L1)[0]
                    if lt.item() == lt.item():
                        rloss3 += lt

# -----------------------------------------------------------------------------
# Backpropagate all losses and update the optimizer
# -----------------------------------------------------------------------------
        loss = Uloss + sloss + sloss1
        if rloss3 == rloss3 and rloss3 > 0:
            loss += rloss3 * 10
        if len(kloss) == 0:
            kloss = 0
        else:
            kloss = torch.mean(torch.cat(kloss))
            kk += kloss.item()
        if kloss == kloss:
            loss += kloss * 10
        tr_loss += loss.item()

        loss.backward()

        oU.step()
        oU.zero_grad()
        oEX.step()
        oEX.zero_grad()
        oEN.step()
        oEN.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f},KK:{:.4f}'.format(
        epoch_loss, kk / len(data_loader_train)))

# -----------------------------------------------------------------------------
# Validation phase:
# - Similar to the training phase but with some simplifications.
# - Includes model saving at the end.
# -----------------------------------------------------------------------------
    if epoch >= 1:

        tk0 = tqdm(data_loader_val, desc="Iteration")

        U.eval()
        EN.eval()
        EX.eval()

        val_loss = 0
        kk = 0
        se = 0
        iou = 0
        n = 0
        for step, batch in enumerate(tk0):
            inputs = batch["image"].unsqueeze(1)

            labels = batch["labels"]
            in2 = batch["im2"].unsqueeze(1)
            la2 = batch["la2"]
            inputs = inputs[:, :, win2[0]:win2[0] +
                            256, win2[1]:win2[1] +
                            256, win2[2]:win2[2] +
                            32].to(device, dtype=torch.float)
            labels = labels[:, win2[0]:win2[0] +
                            256, win2[1]:win2[1] +
                            256, win2[2]:win2[2] +
                            32].to(device, dtype=torch.float)
            in2 = in2[:, :, win2[0]:win2[0] + 256, win2[1]:win2[1] +
                      256, win2[2]:win2[2] + 32].to(device, dtype=torch.float)
            la2 = la2[:, win2[0]:win2[0] + 256, win2[1]:win2[1] +
                      256, win2[2]:win2[2] + 32].to(device, dtype=torch.float)

            p1 = batch["p1"].unsqueeze(1)[:, :,
                win2[0]:win2[0] + 256,
                win2[1]:win2[1] + 256,
                win2[2]:win2[2] + 32].to( device,
                dtype=torch.float)
            p2 = batch["p2"].unsqueeze(1)[:, :,
                win2[0]:win2[0] + 256,
                win2[1]:win2[1] + 256,
                win2[2]:win2[2] + 32].to( device,
                dtype=torch.float)
            s1 = batch["s1"][:, win2[0]:win2[0] +
                             256, win2[1]:win2[1] +
                             256, win2[2]:win2[2] +
                             32].to(device, dtype=torch.float)
            s2 = batch["s2"][:, win2[0]:win2[0] +
                             256, win2[1]:win2[1] +
                             256, win2[2]:win2[2] +
                             32].to(device, dtype=torch.float)
            lsize1 = batch["size1"][:, win2[0]:win2[0] + 256,
                                    win2[1]:win2[1] + 256, win2[2]:win2[2] + 32]
            lx2 = torch.unique(p1[0, 0]).shape[0]
            s4 = batch["s4"][:, win2[0]:win2[0] +
                             256, win2[1]:win2[1] +
                             256, win2[2]:win2[2] +
                             32].to(device, dtype=torch.float)
            in3 = batch["img3"].unsqueeze(1)[:, :,
                win2[0]:win2[0] + 256,
                win2[1]:win2[1] + 256,
                win2[2]:win2[2] + 32].to( device,
                dtype=torch.float)
            pl1 = batch["p1"].unsqueeze(1)[:, :,
                win2[0]:win2[0] + 256,
                win2[1]:win2[1] + 256,
                win2[2]:win2[2] + 32].to( device,
                dtype=torch.float)
       
            pl2 = batch["p2"].unsqueeze(1)[:, :,
                win2[0]:win2[0] + 256,
                win2[1]:win2[1] + 256,
                win2[2]:win2[2] + 32].to( device,
                dtype=torch.float)
            ee1 = list((pl1[0, 0][pl1[0, 0] != 0]).flatten().cpu().numpy())
            ee2 = list((pl2[0, 0][pl2[0, 0] != 0]).flatten().cpu().numpy())
            with torch.no_grad():
                Uout, uo, fo, zs1, size1 = U(PA(torch.cat([inputs, in2], 1)))
                Uout2, uo2, fo2, zs2, size2 = U(PA(torch.cat([in2, in3], 1)))
                po = (F.sigmoid(zs1[:, 0]) > 0.5) * 100 + \
                    (F.sigmoid(zs1[:, 1]) > 0.5) * 200
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
                uo1 = uo.argmax(1)
                uc = ((search_mask(uo[:, :].max(1)[0]).cuda().max(1)[
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
                uo1 = uo2.argmax(1)

                uc = ((search_mask(uo2[:, :].max(1)[0]).cuda().max(1)[
                      0] == uo2[:, 4])) * (Uout2.argmax(1) == 1)
                nloss1 = crloss(uo, (zp1).long())
                nloss2 = crloss((uo2), (zp2).long())
                nloss3 = torch.mean(nloss1[labels > 0])
                nloss4 = torch.mean(nloss2[la2 > 0])
                Uloss = torch.mean(nloss1) + \
                    torch.mean(nloss2) + nloss3 + nloss4

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

                xpp1 = p1.clone()

                xpp2 = p2.clone()

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

            loss = Uloss * 0.01
            if rloss3 == rloss3:
                loss += rloss3
            if len(kloss) == 0:
                kloss = 0
            else:
                kloss = torch.mean(torch.cat(kloss))
                kk += kloss.item()
            if kloss == kloss:
                loss += kloss * 10
            val_loss += loss.item()

            se += Uloss.item()

            n = n + 1
        sU.step(se)
        sEX.step(kk)
        sEN.step(kk)

        epoch_loss = val_loss / len(data_loader_val)
# -----------------------------------------------------------------------------
# Save models
# -----------------------------------------------------------------------------
        print(
            'val Loss: {:.4f},uloss:{:.4f},kloss:{:.4f}'.format(
                np.mean(epoch_loss),
                np.mean(
                    se /
                    len(data_loader_val)),
                np.mean(
                    kk /
                    len(data_loader_val))))
        if epoch > 2:
            va = epoch_loss
            if not os.path.exists(mp):
                os.makedirs(mp, exist_ok=True)

            torch.save(EX.state_dict(), mp +
                       '/EX+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, va))
            torch.save(EN.state_dict(), mp +
                       '/EN+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, va))
            torch.save(U.state_dict(), mp +
                       '/U-ext+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, va))
