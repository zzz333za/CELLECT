
import numpy as np
import torch

from torch import nn



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
                return ey,epy,esp,ezsp
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