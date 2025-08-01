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
from recoloss import CrossEntropyLabelSmooth, TripletLoss, crloss, validation_simple, sort_feature, sort_labels
from recoloss import kflb as search_mask
from recoloss import ud as z_enlarge
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
parser.add_argument('--processed_data_dir', type=str, required=True,
                    help="Path to the preprocessed data directory")
parser.add_argument('--train', type=str, required=True, help="Training data id")

parser.add_argument('--val', type=str, required=True, help="Validation data id")
parser.add_argument('--model_dir', type=str, required=True,
                    help="Path to the output models directory")

args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.processed_data_dir
model_dir = args.model_dir
training_dir = int(args.train)
validation_dir = int(args.val)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
######Training dataset setup (Global)
training_dir_str = str(training_dir)
train_tif_files = os.listdir(data_dir + '/mskcc_confocal_s' + training_dir_str + '/images/')
train_tif_files = [i for i in train_tif_files if 'tif' in i]

train_tif_dict = {}
train_output_npy_files = os.listdir(output_dir + '/ls' + training_dir_str + '/')
for i in tqdm(train_tif_files):
    if 'tif' in i:

        frame_index = int(i.split('_')[-1].split('.')[0][1:])
        if frame_index < 275:
            train_tif_dict[frame_index] = data_dir + '/mskcc_confocal_s' + training_dir_str + '/images/' + i

train_tracks_df = pd.read_table(data_dir + '/mskcc_confocal_s' + training_dir_str + '/tracks/tracks.txt')
train_polar_bodies_df = pd.read_table(data_dir + '/mskcc_confocal_s' + training_dir_str + '/tracks/tracks_polar_bodies.txt')


######Validation dataset setup (Global)
validation_dir_str = str(validation_dir)
val_tif_files = os.listdir(data_dir + '/mskcc_confocal_s' + validation_dir_str  + '/images/')
val_tif_files = [i for i in val_tif_files if 'tif' in i]
val_output_npy_files = os.listdir(output_dir + '/ls' + validation_dir_str  + '/')
val_tif_dict = {}
for i in tqdm(val_tif_files):
    if 'tif' in i:

        num = int(i.split('_')[-1].split('.')[0][1:])
        if num < 275:
            val_tif_dict[num] = data_dir + '/mskcc_confocal_s' + validation_dir_str  + '/images/' + i

val_tracks_df = pd.read_table(data_dir + '/mskcc_confocal_s' + validation_dir_str  + '/tracks/tracks.txt')
val_polar_bodies_df = pd.read_table(data_dir + '/mskcc_confocal_s' + validation_dir_str  + '/tracks/tracks_polar_bodies.txt')

val_ymin = int(val_tracks_df.y.min() - 30)
val_ymax = max(int(val_tracks_df.y.max() + 30), val_ymin + 256)
val_xmin = int(val_tracks_df.x.min() - 30)
val_xmax = max(val_xmin + 256, int(val_tracks_df.x.max() + 30))
val_crop_offset = [min(max(0, (val_ymax - val_ymin - 256) // 2), 30),
                   min(max(0, (val_xmax - val_xmin - 256) // 2), 30), 2]


# -----------------------------------------------------------------------------
# Configure the data loader
# -----------------------------------------------------------------------------
class TrainingDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data) - 3 + 100

    def __getitem__(self, index):

        if index  < 268:

            frame_id = index
        else:
            frame_id = random.choice(list(range(250, 272, 1)))

        img_path_t = train_tif_dict[frame_id]
        offset = 1
        img_path_t1 = train_tif_dict[frame_id + offset]
        img_path_t2	 = train_tif_dict[frame_id + offset + offset]
        training_dir_str = str(training_dir)

        img_t = tifffile.TiffFile(img_path_t).asarray().transpose([1, 2, 0])
        img_t1 = tifffile.TiffFile(img_path_t1).asarray().transpose([1, 2, 0])
        img_t2 = tifffile.TiffFile(img_path_t2).asarray().transpose([1, 2, 0])
        mask_k1 =  np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k1-3d-1-imaris.npy')
        mask_k2 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k2-3d-1-imaris.npy')
        mask_k5 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k5-3d-1-imaris.npy')
        mask_k6 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k6-3d-1-imaris.npy')
        label_k1 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k1-3d-imaris.npy')
        label_k2 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k2-3d-imaris.npy')
        label_k3 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k3-3d-imaris.npy')
        label_k4 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k4-3d-imaris.npy')
        mask_k6_next = np.load(output_dir + '/ls' + training_dir_str + '/' +
                      str(frame_id + 1) + '-k6-3d-1-imaris.npy')
        size_map1 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k31-imaris.npy')
        size_map2 = np.load(output_dir + '/ls' + training_dir_str + '/' + str(frame_id ) + '-k32-imaris.npy')
        crop_x, crop_y, crop_z = random.randint(50, max(0, mask_k1.shape[0] - 256)), random.randint(
            50, max(0, mask_k1.shape[1] - 256)), random.randint(3, max(0, mask_k1.shape[2] - 32))
        crop_height, crop_width, crop_depth = 256, 256, 32
        img_t_tensor = torch.from_numpy((img_t.astype(float)))
        mask_k1_tensor = torch.from_numpy(mask_k1)
        img_t2_tensor= torch.from_numpy(img_t2.astype(float))
        img_t1_tensor = torch.from_numpy(img_t1.astype(float))
        mask_k2_tensor = torch.from_numpy(mask_k2)
        mask_k5_tensor = torch.from_numpy(mask_k5)
        mask_k6_tensor = torch.from_numpy(mask_k6)
        mask_k6_next_tensor = torch.from_numpy(mask_k6_next)
        label_k1_tensor = torch.from_numpy(label_k1)
        label_k2_tensor = torch.from_numpy(label_k2)
        label_k3_tensor = torch.from_numpy(label_k3)
        label_k4_tensor = torch.from_numpy(label_k4)
        size_map1_tensor  = torch.from_numpy(size_map1)
        size_map2_tensor = torch.from_numpy(size_map2)
        return {'image_t': img_t_tensor[crop_x:crop_x + crop_height,
                         crop_y:crop_y + crop_width,
                         crop_z:crop_z + crop_depth],
                'label_k1': label_k1_tensor[crop_x:crop_x + crop_height,
                          crop_y:crop_y + crop_width,
                          crop_z:crop_z + crop_depth],
                'image_t1': img_t1_tensor[crop_x:crop_x + crop_height,
                       crop_y:crop_y + crop_width,
                       crop_z:crop_z + crop_depth],
                'label_k2': label_k2_tensor[crop_x:crop_x + crop_height,
                       crop_y:crop_y + crop_width,
                       crop_z:crop_z + crop_depth],
                'mask_k1': mask_k1_tensor[crop_x:crop_x + crop_height,
                      crop_y:crop_y + crop_width,
                      crop_z:crop_z + crop_depth],
                'mask_k2': mask_k2_tensor[crop_x:crop_x + crop_height,
                      crop_y:crop_y + crop_width,
                      crop_z:crop_z + crop_depth],
                'division_mask_k5': mask_k5_tensor[crop_x:crop_x + crop_height,
                      crop_y:crop_y + crop_width,
                      crop_z:crop_z + crop_depth],
                'division_mask_k6': mask_k6_tensor[crop_x:crop_x+ crop_height,
                      crop_y:crop_y + crop_width,
                      crop_z:crop_z + crop_depth],
                'division_mask_k6_next': mask_k6_next_tensor[crop_x:crop_x + crop_height,
                      crop_y:crop_y + crop_width,
                      crop_z:crop_z + crop_depth],
                'label_k3': label_k3_tensor[crop_x:crop_x + crop_height,
                       crop_y:crop_y + crop_width,
                       crop_z:crop_z + crop_depth],
                'label_k4': label_k4_tensor[crop_x:crop_x + crop_height,
                       crop_y:crop_y + crop_width,
                       crop_z:crop_z + crop_depth],
                'size_map1': size_map1_tensor[crop_x:crop_x + crop_height,
                         crop_y:crop_y + crop_width,
                         crop_z:crop_z + crop_depth],
                'size_map2': size_map2_tensor[crop_x:crop_x + crop_height,
                         crop_y:crop_y + crop_width,
                         crop_z:crop_z + crop_depth],
                'image_t2': img_t2_tensor[crop_x:crop_x + crop_height,
                        crop_y:crop_y + crop_width,
                        crop_z:crop_z + crop_depth]}


class ValidationDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 203

    def __getitem__(self, index):
        frame_id = index + 200
        img_path_t = val_tif_dict[frame_id]
        offset = 1
        img_path_t1 = val_tif_dict[frame_id + offset]
        img_path_t2 = val_tif_dict[frame_id + offset + offset]
        validation_dir_str = str(validation_dir)
        img_t = tifffile.TiffFile(img_path_t).asarray().transpose([1, 2, 0])
        img_t1 = tifffile.TiffFile(img_path_t1).asarray().transpose([1, 2, 0])
        img_t2 = tifffile.TiffFile(img_path_t2).asarray().transpose([1, 2, 0])
        mask_k1 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k1-3d-1-imaris.npy')
        mask_k2 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k2-3d-1-imaris.npy')
        mask_k5 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k5-3d-1-imaris.npy')
        mask_k6 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k6-3d-1-imaris.npy')
        label_k1 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k1-3d-imaris.npy')
        label_k2 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k2-3d-imaris.npy')
        mask_k6_next = np.load(output_dir + '/ls' + validation_dir_str + '/' +
                      str(frame_id + 1) + '-k6-3d-1-imaris.npy')
        size_map1 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k31-imaris.npy')
        size_map2 = np.load(output_dir + '/ls' + validation_dir_str + '/' + str(frame_id) + '-k32-imaris.npy')


        img_t_tensor = torch.from_numpy(img_t.astype(float))
        img_t1_tensor = torch.from_numpy(img_t1.astype(float))
        img_t2_tensor = torch.from_numpy(img_t2.astype(float))
        mask_k1_tensor = torch.from_numpy(mask_k1)
        mask_k2_tensor = torch.from_numpy(mask_k2)
        mask_k5_tensor = torch.from_numpy(mask_k5)
        mask_k6_tensor = torch.from_numpy(mask_k6)
        mask_k6_next_tensor = torch.from_numpy(mask_k6_next)
        label_k1_tensor = torch.from_numpy(label_k1)
        label_k2_tensor = torch.from_numpy(label_k2)
        size_map1_tensor = torch.from_numpy(size_map1)
        size_map2_tensor = torch.from_numpy(size_map2)

        return {
            'image_t': img_t_tensor,
            'label_k1': label_k1_tensor,
            'image_t1': img_t1_tensor,
            'label_k2': label_k2_tensor,
            'mask_k1': mask_k1_tensor,
            'mask_k2': mask_k2_tensor,
            'division_mask_k5': mask_k5_tensor,
            'division_mask_k6': mask_k6_tensor,
            'division_mask_k6_next': mask_k6_next_tensor,
            'image_t2': img_t2_tensor,
            'size_map1': size_map1_tensor,
            'size_map2': size_map2_tensor
        }


# -----------------------------------------------------------------------------
# Set training environment parameters
# -----------------------------------------------------------------------------
SMOOTH = 1e-6
n_epochs = 150
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0")
device1 = torch.device("cpu")

# Temporal kernel generation
temporal_padding_xy = 2
center_kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 2, 1, 0, 0],
                          [0, 1, 2, 3, 2, 1, 0],
                          [1, 2, 3, 4, 3, 2, 1],
                          [0, 1, 2, 3, 2, 1, 0],
                          [0, 0, 1, 2, 1, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]])
center_kernel = torch.from_numpy(center_kernel).float().to(device).reshape([1, 1, 7, 7, 1])


feature_extract_net = UNet3D(2, 6)
inter_frame_matcher = EXNet(64, 8) 
intra_frame_matcher = EXNet(64, 6) 

feature_extract_net.to(device)
inter_frame_matcher.to(device)
intra_frame_matcher.to(device)

plist = [{'params': feature_extract_net.parameters()}]
optimizer_seg = optim.Adam(plist, lr=2e-4)
plist = [{'params': inter_frame_matcher.parameters(), 'lr': 2e-4}]
optimizer_inter = optim.Adam(plist)
plist = [{'params': intra_frame_matcher.parameters(), 'lr': 2e-4}]
optimizer_intra = optim.Adam(plist)

###Loss Function
weights_2class = torch.tensor([1.0, 32.0]).cuda()
loss_segmentation_2cls = torch.nn.CrossEntropyLoss(reduction='none', weight=weights_2class)

weights_3cls = torch.tensor([1.0, 32.0, 1.0]).cuda()
loss_segmentation_3cls = torch.nn.CrossEntropyLoss(reduction='none', weight=weights_3cls)

weights_5cls = torch.tensor([1.0, 32.0, 32, 32, 32]).cuda()
loss_segmentation_5cls = torch.nn.CrossEntropyLoss(reduction='none', weight=weights_5cls)

weights_division = torch.tensor([1.0, 1]).cuda()
loss_division = torch.nn.CrossEntropyLoss(reduction='none', weight=weights_division)

loss_segmentation_plain = torch.nn.CrossEntropyLoss(reduction='none')


loss_margin_ranking = torch.nn.MarginRankingLoss(margin=0.3)
loss_triplet = TripletLoss(0.3)
loss_l1 = torch.nn.L1Loss(reduction='none')
loss_mse = torch.nn.MSELoss()
loss_bce = torch.nn.BCELoss(reduction='none')

scheduler_segmentation = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, factor=0.5, patience=3)
scheduler_inter_frame = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_inter, factor=0.5, patience=3)
scheduler_intra_frame = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_intra, factor=0.5, patience=3)

print('    Total params: %.2fM' % (sum(p.numel()
                                       for p in feature_extract_net.parameters()) / 1000000.0))

train_loss_history = []
val_loss_history = []
train_dataset = TrainingDataset(train_tif_dict)
data_loader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = ValidationDataset(val_tif_dict)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=0)

best_val_loss = 1000
val_score_record = 10000

# -----------------------------------------------------------------------------
# Data preprocessing (PA)
# -----------------------------------------------------------------------------
def transform(x):
    x = torch.log1p(x)
    return x


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    feature_extract_net.train()
    if epoch < 6 or (epoch > 15 and epoch <= 20):
        inter_frame_matcher.train()
        intra_frame_matcher.train()

    else:
        inter_frame_matcher.eval()
        intra_frame_matcher.eval()

    train_loss_sum = 0
    train_kloss_sum = 0

    train_progress_bar = tqdm(data_loader_train, desc="Iteration")
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
    for step, batch in enumerate(train_progress_bar):
        # -----------------------------------------------------------------------------
        # Loading
        # -----------------------------------------------------------------------------
        image_t  = batch["image_t"].unsqueeze(1)
        label_k1 = batch["label_k1"]
        image_t1 = batch["image_t1"].unsqueeze(1)
        label_k2 = batch["label_k2"]
        label_k3 = batch["label_k3"]
        label_k4 = batch["label_k4"]

        image_t = image_t.to(device, dtype=torch.float)
        label_k1 = label_k1.to(device, dtype=torch.float)
        image_t1  = image_t1 .to(device, dtype=torch.float)
        label_k2 = label_k2.to(device, dtype=torch.float)
        label_k3 = label_k3.to(device, dtype=torch.float)
        label_k4 = label_k4.to(device, dtype=torch.float)

        center_seg_1 = batch["mask_k1"].to(device, dtype=torch.float).unsqueeze(1)
        center_seg_2 = batch["mask_k2"].to(device, dtype=torch.float).unsqueeze(1)

        center_mask_1 = batch["mask_k1"].to(device, dtype=torch.float).unsqueeze(1)
        center_mask_2 = batch["mask_k2"].to(device, dtype=torch.float).unsqueeze(1)

        edge_ids_1 = list((center_mask_1[0, 0][center_mask_1[0, 0] != 0]).flatten().cpu().numpy())
        edge_ids_2 = list((center_mask_2[0, 0][center_mask_2[0, 0] != 0]).flatten().cpu().numpy())
        unique_label_count = torch.unique(center_mask_1[0, 0]).shape[0]
        is_complex_match = (unique_label_count > 400) * 1

        gt_size_1 = batch["size_map1"].to(device, dtype=torch.float)
        gt_size_2 = batch["size_map2"].to(device, dtype=torch.float)

        division_map_1 = batch["division_mask_k5"].to(device, dtype=torch.float)
        division_map_2 = batch["division_mask_k6"].to(device, dtype=torch.float)
        division_map_3 = batch["division_mask_k6_next"].to(device, dtype=torch.float)

        image_t2 = batch["image_t2"].unsqueeze(1).to(device, dtype=torch.float)
        # -----------------------------------------------------------------------------
        # Run the main model to obtain outputs for:
        # - Cell segmentation
        # - Center point scoring
        # - Division likelihood assessment
        # - Size estimation
        # -----------------------------------------------------------------------------
        seg_1, score_map_1, feature_map_1, division_pred_1, size_pred_1 \
            = feature_extract_net(transform(torch.cat([image_t, image_t1], 1)))
        seg_2, score_map_2, feature_map_2, division_pred_2, size_pred_2 \
            = feature_extract_net(transform(torch.cat([image_t1, image_t2], 1)))

        center_seg_1[:, 0] = F.conv3d((center_seg_1[:, 0] > 0).float().unsqueeze(
            1), center_kernel, padding=(temporal_padding_xy + 1, temporal_padding_xy + 1, 0))
        center_seg_2[:, 0] = F.conv3d((center_seg_2[:, 0] > 0).float().unsqueeze(
            1), center_kernel, padding=(temporal_padding_xy + 1, temporal_padding_xy + 1, 0))

        for i in range(3):
            # Expand segmentation along Z
            center_seg_1 = z_enlarge(center_seg_1)
            center_seg_2 = z_enlarge(center_seg_2)

        original_division_map = division_map_1.clone()
        division_map_1 = torch.cat([(division_map_1 == 1).float(), (division_map_2 == 2)], 0)

        division_map_2 = torch.cat([(division_map_2 == 1).float(), (division_map_3 == 2)], 0)

        label_k3 = (label_k1 > 0).float() + 2 * (label_k3 > 0).float()
        label_k3[label_k3 > 2] = 2
        label_k4 = (label_k2 > 0).float() + 2 * (label_k4 > 0).float()
        label_k4[label_k4 > 2] = 2

        weighted_mask1 = label_k3 > 0
        weighted_zmask1 = weighted_mask1 * (center_seg_1[:, 0])
        weighted_mask2 = label_k4 > 0
        weighted_zmask2 = weighted_mask2 * (center_seg_2[:, 0])

        # -----------------------------------------------------------------------------
        # Segmentation loss
        # -----------------------------------------------------------------------------
        loss_seg_1 = loss_segmentation_3cls(seg_1, (label_k3).long())
        loss_seg_2 = loss_segmentation_3cls(seg_2, (label_k4).long())


        label_k1_for_loss = label_k1.clone()
        label_k2_for_loss = label_k2.clone()

        valid_mask1 = label_k1_for_loss > 0
        valid_mask2 = label_k2_for_loss > 0

        # -----------------------------------------------------------------------------
        # Size loss
        # -----------------------------------------------------------------------------
        size_loss_1 = loss_mse(size_pred_1, gt_size_1)
        size_loss_2 = loss_mse(size_pred_2, gt_size_2)
        total_size_loss = torch.mean(size_loss_1) + torch.mean(size_loss_2)

        # -----------------------------------------------------------------------------
        # Division loss
        # -----------------------------------------------------------------------------
        division_loss = torch.mean((loss_bce(F.sigmoid(division_pred_1),
                                    division_map_1.unsqueeze(0)) * (1 +  division_map_1 * 15)))
        division_loss = division_loss * (division_map_1.sum() > 0)

        # ---------------------------------------------------------------------
        # Feature-based loss for cell embedding, matching and contrastive tasks
        # Loss computation based on given cell points
        # Extract information for each individual cell
        # ---------------------------------------------------------------------
        contrastive_loss = 0
        classification_loss_list = []


        feature_map_t  = feature_map_1.transpose(0, 1)
        feature_map_t1 = feature_map_2.transpose(0, 1)
        # -----------------------------------------------------------------------------
        # Center point extraction
        # -----------------------------------------------------------------------------
        center_pred_mask = ((search_mask(score_map_1[:, :].max(1)[0]).cuda().max(1)[0] ==
               score_map_1[:, 4])) * (seg_1.argmax(1) == 1)
        center_indices = torch.where(center_pred_mask.to(device, dtype=torch.float))


        features_center = feature_map_t[:, center_indices[0], center_indices[1], center_indices[2],
                          center_indices[3]].transpose(0, 1)
        size_center = size_pred_1[center_indices[0], 0, center_indices[1], center_indices[2], center_indices[3]]
        division_scores_center = F.sigmoid(division_pred_1)[center_indices[0], :, center_indices[1], center_indices[2],
                                 center_indices[3]]

        center_labels_1 = label_k1[center_indices[0], center_indices[1], center_indices[2], center_indices[3]]
        division_targets_binary = original_division_map[
                                      center_indices[0], center_indices[1], center_indices[2], center_indices[3]] == 1
        center_pred_mask_next = ((search_mask(score_map_2[:, :].max(1)[0]).cuda().max(1)[0] ==
                                  score_map_2[:, 4]) * (seg_2.argmax(1) == 1))

        for cell_id in edge_ids_1:
            if edge_ids_2.count(cell_id) > 1 :
                label_k1_for_loss[label_k1_for_loss == cell_id] = 4
                label_k2_for_loss[label_k2_for_loss == cell_id] = 4

        label_k1_for_loss[label_k1_for_loss != 4] = 1
        label_k2_for_loss[label_k2_for_loss != 4] = 1


        # ----------------------------------------
        # Loss from center position classification maps
        # ----------------------------------------
        loss_center_1 = crloss(score_map_1, (weighted_zmask1).long()) * label_k1_for_loss
        loss_center_2 = crloss(score_map_2, (weighted_zmask2).long()) * label_k2_for_loss
        loss_center_mean_1 = torch.mean(loss_center_1[label_k1 > 0])
        loss_center_mean_2 = torch.mean(loss_center_2[label_k2 > 0])

        loss_seg_total = (torch.mean(loss_center_1) +
                             torch.mean(loss_center_2) +
                             loss_center_mean_1 +
                             loss_center_mean_2 +
                             torch.mean(loss_seg_1 * label_k1_for_loss) +
                             torch.mean(loss_seg_2 * label_k2_for_loss))

        loss_seg_total *= (1 + is_complex_match)

        # -----------------------------------------------------------------------------
        # All subsequent losses are computed on a per-cell basis (cell-wise).
        # -----------------------------------------------------------------------------
        if center_indices[0].shape[0] > 0 and epoch > 1:

            center_indices_next = torch.where(center_pred_mask_next.to(device, dtype=torch.float))
            feature_vector_next = feature_map_t1[:, center_indices_next[0], center_indices_next[1],
                                  center_indices_next[2],
                                  center_indices_next[3]].transpose(0, 1)
            cell_size_next = size_pred_2[center_indices_next[0], 0, center_indices_next[1], center_indices_next[2],
            center_indices_next[3]]
            division_prob_next = F.sigmoid(division_pred_2)[center_indices_next[0], :, center_indices_next[1],
                                 center_indices_next[2], center_indices_next[3]]
            label_values_next = label_k2[center_indices_next[0], center_indices_next[1],
            center_indices_next[2], center_indices_next[3]]
            coord_current = torch.stack(center_indices[1:], dim=1)
            coord_next = torch.stack(center_indices_next[1:], dim=1)
            # resolution xy/z=5
            coord_current[:, 2] *= 5
            coord_next[:, 2] *= 5

            coords_1 = coord_current.float()
            coords_2 = coord_next.float()

            # -----------------------------------------------------------------------------
            # Set an error tolerance range to match the extracted center points
            # with the sparsely annotated center points
            # Label sorting for matching
            # -----------------------------------------------------------------------------

            features_1, labels_1, coords_1, features_2, labels_2, coords_2 = \
                features_center, center_labels_1, coords_1, feature_vector_next, label_values_next, coords_2

            if features_1.shape[0] > 5 and features_2.shape[0] > 5:
                # -----------------------------------------------------------------------------
                # frame 1

                labels_1 = sort_labels(coords_1, center_mask_1, labels_1, edge_ids_1, ratio=5)

                # -----------------------------------------------------------------------------
                # frame 2

                labels_2 = sort_labels(coords_2, center_mask_2, labels_2, edge_ids_2, ratio=5)

                # -----------------------------------------------------------------------------
                # Train the MLP model to distinguish identical cells within the
                # same frame (Intra-frame classification)
                # -----------------------------------------------------------------------------

                num_points = coords_1.shape[0]
                dist_matrix = torch.pow(coords_1.float(), 2).sum(dim=1, keepdim=True).expand(num_points, num_points) + \
                              torch.pow(coords_1.float(), 2).sum(dim=1, keepdim=True).expand(num_points, num_points).t()
                dist_matrix.addmm_(1, -2, coords_1.float(), coords_1.float().t())
                dist_values, dist_indices = dist_matrix.topk(6, largest=False)
                dist_indices = dist_indices[:, 1:]
                dist_values = dist_values[:, 1:]

                if random.randint(0, 1) == 1:
                    new_indices = dist_indices.clone()
                    new_values = dist_values.clone()
                    for idx in range(dist_indices.shape[0]):
                        perm = list(range(5))
                        random.shuffle(perm)
                        for jdx in range(5):
                            new_indices[idx, jdx] = dist_indices[idx, perm[jdx]]
                            new_values[idx, jdx] = dist_values[idx, perm[jdx]]
                    dist_indices = new_indices
                    dist_values = new_values

                features_k1, positions_k1, size_k1, division_k1 = \
                    sort_feature(features_1, coords_1, coords_1, dist_indices,
                                 size_center, size_center, division_scores_center, division_scores_center, n=5)

                score_intra = intra_frame_matcher(features_1.unsqueeze(1),
                                                  features_k1, positions_k1.float(),
                                                  size_k1, division_k1)

                match_labels_list = []
                total_matches = 0

                for i in range(5):
                    match_labels_list.append((labels_1[dist_indices[:, i]] == labels_1).unsqueeze(1))
                    total_matches += (labels_1[dist_indices[:, i]] == labels_1).float()

                match_labels_list.append((total_matches == 0).unsqueeze(1))
                match_labels = torch.cat(match_labels_list, 1)

                score_intra = score_intra[labels_1 != 0]
                match_labels = match_labels[labels_1 != 0]

                intra_loss = torch.mean(loss_bce(F.sigmoid(score_intra), match_labels.float()))

                # -----------------------------------------------------------------------------
                # Train the MLP model to distinguish identical cells across different frames
                # and identify divided cells.
                # The difference compared to another model lies in whether divided cells
                # are considered the same cell.  (inter-frame matching)
                # -----------------------------------------------------------------------------
                coords_1_cpu = coords_1.to(device1)
                coords_2_cpu = coords_2.to(device1)
                label_values_1_np = labels_1.cpu().numpy()

                division_count_weights = np.array([edge_ids_2.count(lbl) for lbl in label_values_1_np])
                division_count_weights = (division_count_weights > 1) * 9 + 1

                dist_matrix_inter = torch.cdist(coords_1_cpu, coords_2_cpu, p=2)
                dist_values_inter, dist_indices_inter = dist_matrix_inter.topk(5, largest=False)

                coords_1 = coords_1.to(device)
                coords_2 = coords_2.to(device)

                if random.randint(0, 1) == 1:
                    new_indices_inter = dist_indices_inter.clone()
                    new_values_inter = dist_values_inter.clone()
                    for idx in range(dist_indices_inter.shape[0]):
                        perm = list(range(5))
                        random.shuffle(perm)
                        for jdx in range(5):
                            new_indices_inter[idx, jdx] = dist_indices_inter[idx, perm[jdx]]
                            new_values_inter[idx, jdx] = dist_values_inter[idx, perm[jdx]]
                    dist_indices_inter = new_indices_inter
                    dist_values_inter = new_values_inter

                features_k2, positions_k2, size_k2, division_k2 = sort_feature(
                    features_2, coords_2, coords_1, dist_indices_inter,
                    cell_size_next, size_center,
                    division_prob_next, division_scores_center, n=5)

                score_inter = inter_frame_matcher(
                    features_1.unsqueeze(1),
                    features_k2, positions_k2.float(),
                    size_k2, division_k2)

                division_logits_inter = score_inter[:, -2:]
                score_inter = score_inter[:, :-2]

                match_labels_list_inter = []
                total_matches_inter = 0

                for i in range(5):
                    match_labels_list_inter.append(
                        (labels_2[dist_indices_inter[:, i]] == labels_1).unsqueeze(1))
                    total_matches_inter += (labels_2[dist_indices_inter[:, i]] == labels_1).float()

                match_labels_list_inter.append((total_matches_inter == 0).unsqueeze(1))
                match_labels_inter = torch.cat(match_labels_list_inter, 1)

                valid_mask_inter = labels_1 != 0
                score_inter = score_inter[valid_mask_inter]
                match_labels_inter = match_labels_inter[valid_mask_inter]
                division_count_weights = division_count_weights[valid_mask_inter.cpu()]

                inter_classification_loss_vec = loss_bce(F.sigmoid(score_inter), match_labels_inter.float())
                inter_classification_loss_vec[torch.tensor(division_count_weights > 1)] *= 4

                inter_classification_loss = torch.mean(inter_classification_loss_vec)
                division_loss_center_prediction = loss_division(division_logits_inter, division_targets_binary.long())

                classification_total_loss = intra_loss + inter_classification_loss * 5 + 5 * torch.mean(
                    division_loss_center_prediction)

                if classification_total_loss.item() == classification_total_loss.item():
                    classification_loss_list.append((classification_total_loss).unsqueeze(-1))

            # -----------------------------------------------------------------------------
            # Contrastive learning phase
            # Obtain equal amounts of data pairs for each class across two frames:
            # - For the same cell, one sample per frame (total of 2 samples).
            # - For divided cells, three samples per instance.
            # -----------------------------------------------------------------------------
            label_1_all = labels_1
            label_2_all = labels_2

            label_1_list = list((label_1_all).cpu().numpy())
            label_count_1 = pd.value_counts(label_1_list).to_dict()
            label_2_list = list((label_2_all).cpu().numpy())
            label_count_2 = pd.value_counts(label_2_list).to_dict()
            label_1_np = label_1_all.cpu().numpy().astype(int)
            label_2_np = label_2_all.cpu().numpy().astype(int)

            index_valid_1 = []
            id_seen_1 = []

            for i in range(len(label_1_all)):
                if label_1_np[i] != 0 and label_count_1.get(label_1_np[i], 0) >= 1 \
                        and label_count_2.get(label_1_np[i], 0) >= 2 and label_1_np[i] not in id_seen_1:
                    index_valid_1.append(i)
                    id_seen_1.append(label_1_np[i])
            index_valid_1 = np.array(index_valid_1)
            features_1_div = features_1[index_valid_1]
            coords_1_div = coords_1[index_valid_1]
            labels_1_div = label_1_all[index_valid_1]

            index_valid_1 = []
            id_seen_1 = []

            for i in range(len(label_1_all)):
                if label_1_np[i] != 0 and label_count_1.get(label_1_np[i], 0) >= 1 \
                        and label_count_2.get(label_1_np[i], 0) >= 1 and label_1_np[i] not in id_seen_1:
                    index_valid_1.append(i)
                    id_seen_1.append(label_1_np[i])

            index_valid_1 = np.array(index_valid_1)
            features_1_common = features_1[index_valid_1]
            coords_1_common = coords_1[index_valid_1]
            labels_1_common = label_1_all[index_valid_1]

            label_2_np = label_2_all.cpu().numpy()
            index_valid_2a = []
            index_valid_2b = []
            id_seen_2 = []
            for i in range(len(label_2_all)):
                if label_2_np[i] != 0 and label_count_1.get(label_2_np[i], 0) >= 1 and label_count_2.get(label_2_np[i],
                                                                                                         0) >= 2:
                    if label_2_np[i] not in id_seen_2:
                        index_valid_2a.append(i)
                        id_seen_2.append(label_2_np[i])
                    elif id_seen_2.count(label_2_np[i]) < 2:
                        index_valid_2b.append(i)
                        id_seen_2.append(label_2_np[i])
            index_valid_2a = np.array(index_valid_2a)
            index_valid_2b = np.array(index_valid_2b)
            features_2_div_1 = features_2[index_valid_2a]
            coords_2_div_1 = coords_2[index_valid_2a]
            labels_2_div_2 = label_2_all[index_valid_2b]
            features_2_div_2 = features_2[index_valid_2b]
            coords_2_div_2 = coords_2[index_valid_2b]
            labels_2_div_2 = label_2_all[index_valid_2b]

            index_valid_2_common = []
            id_seen_2_common = []

            for i in range(len(label_2_all)):
                if label_2_np[i] != 0 and label_count_1.get(label_2_np[i], 0) >= 1 \
                        and label_count_2.get(label_2_np[i], 0) >= 1 and label_2_np[i] not in id_seen_2_common:
                    index_valid_2_common.append(i)
                    id_seen_2_common.append(label_2_np[i])
            index_valid_2_common = np.array(index_valid_2_common)
            features_2_common = features_2[index_valid_2_common]
            coords_2_common = coords_2[index_valid_2_common]
            labels_2_common = label_2_all[index_valid_2_common]

            triplet_pair_groups = [
                [[features_1_common, features_2_common], [labels_1_common, labels_2_common]],
                [[features_2_div_1, features_2_div_2], [labels_2_div_2, labels_2_div_2]],
                [[features_1_div, torch.cat([features_2_div_1, features_2_div_2], 0)],
                 [labels_1_div, torch.cat([labels_2_div_2, labels_2_div_2], 0)]]
            ]
            # -----------------------------------------------------------------------------
            # Compute triplet loss
            # -----------------------------------------------------------------------------
            for feature_label_pair in triplet_pair_groups:
                feature_list, label_list = feature_label_pair
                concatenated_features = torch.cat(feature_list, 0)
                concatenated_labels = torch.cat(label_list, 0)
                if concatenated_features.shape[0] > 4:
                    triplet_loss_val = loss_triplet(concatenated_features, concatenated_labels)[0]
                    if triplet_loss_val.item() == triplet_loss_val.item():
                        contrastive_loss += triplet_loss_val

        # -----------------------------------------------------------------------------
        # Backpropagate all losses and update the optimizer
        # -----------------------------------------------------------------------------
        total_loss = loss_seg_total + total_size_loss + division_loss
        if contrastive_loss == contrastive_loss and contrastive_loss > 0:
            total_loss += contrastive_loss * 10
        if len(classification_loss_list) == 0:
            classification_loss = 0
        else:
            classification_loss = torch.mean(torch.cat(classification_loss_list))
            train_kloss_sum += classification_loss.item()
        if classification_loss == classification_loss:
            total_loss += classification_loss * 10

        train_loss_sum += total_loss.item()

        total_loss.backward()

        optimizer_seg.step()
        optimizer_seg.zero_grad()
        optimizer_inter.step()
        optimizer_inter.zero_grad()
        optimizer_intra.step()
        optimizer_intra.zero_grad()

    epoch_avg_loss = train_loss_sum / len(data_loader_train)
    print('Training Loss: {:.4f},KK:{:.4f}'.format(
        epoch_avg_loss, train_kloss_sum / len(data_loader_train)))

    # -----------------------------------------------------------------------------
    # Validation phase:
    # - Similar to the training phase but with some simplifications.
    # - Includes model saving at the end.
    # -----------------------------------------------------------------------------
    if epoch >= 1:

        val_progress_bar = tqdm(data_loader_val, desc="Iteration")

        feature_extract_net.eval()
        intra_frame_matcher.eval()
        inter_frame_matcher.eval()

        validation_loss_total = 0
        classification_scheduler = 0
        seg_scheduler = 0
        num_val_batches = 0


        for step, val_batch in enumerate(val_progress_bar):

            input_image_t = val_batch["image_t"].unsqueeze(1)
            label_mask_t = val_batch["label_k1"]
            input_image_t1 = val_batch["image_t1"].unsqueeze(1)
            label_mask_t1 = val_batch["label_k2"]
            input_image_t2 = val_batch["image_t2"].unsqueeze(1)

            center_mask_k1 = val_batch["mask_k1"].unsqueeze(1)
            center_mask_k2 = val_batch["mask_k2"].unsqueeze(1)
            division_mask_k5 = val_batch["division_mask_k5"]
            division_mask_k6 = val_batch["division_mask_k6"]
            division_mask_k6_next = val_batch["division_mask_k6_next"]

            size_target_map_1 = val_batch["size_map1"]
            size_target_map_2 = val_batch["size_map2"]

            # 输入裁剪
            crop_y, crop_x, crop_z = val_crop_offset
            input_image_t = input_image_t[:, :, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(device,
                                                                                                                 dtype=torch.float)
            input_image_t1 = input_image_t1[:, :, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)
            input_image_t2 = input_image_t2[:, :, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)

            label_mask_t = label_mask_t[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(device,
                                                                                                            dtype=torch.float)
            label_mask_t1 = label_mask_t1[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(device,
                                                                                                              dtype=torch.float)

            center_mask_k1 = center_mask_k1[:, :, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)
            center_mask_k2 = center_mask_k2[:, :, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)

            division_mask_k5 = division_mask_k5[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)
            division_mask_k6 = division_mask_k6[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)
            division_mask_k6_next = division_mask_k6_next[:, crop_y:crop_y + 256, crop_x:crop_x + 256,
                                    crop_z:crop_z + 32].to(device, dtype=torch.float)

            size_target_map_1 = size_target_map_1[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)
            size_target_map_2 = size_target_map_2[:, crop_y:crop_y + 256, crop_x:crop_x + 256, crop_z:crop_z + 32].to(
                device, dtype=torch.float)

            seg_loss, contrastive_loss, classification_loss = validation_simple(feature_extract_net, intra_frame_matcher, inter_frame_matcher, 
                                       input_image_t, input_image_t1, input_image_t2, center_mask_k1,
                                       center_mask_k2, label_mask_t, label_mask_t1, division_mask_k5,
                                        division_mask_k6, division_mask_k6_next, transform, step)

            loss = seg_loss * 0.01
            if contrastive_loss == contrastive_loss:
                loss += contrastive_loss
            if len(classification_loss) == 0:
                classification_loss = 0
            else:
                classification_loss = torch.mean(torch.cat(classification_loss))
                classification_scheduler += classification_loss.item()
            if classification_loss == classification_loss:
                loss += classification_loss * 10
            validation_loss_total += loss.item()

            seg_scheduler += seg_loss.item()

        scheduler_segmentation.step(seg_scheduler)
        scheduler_inter_frame.step(classification_scheduler)
        scheduler_intra_frame.step(classification_scheduler)

  
        epoch_loss = validation_loss_total / len(data_loader_val)
        # -----------------------------------------------------------------------------
        # Save models
        # -----------------------------------------------------------------------------
        print(
            'val Loss: {:.4f},uloss:{:.4f},kloss:{:.4f}'.format(
                np.mean(epoch_loss),
                np.mean(
                    seg_loss.item() /
                    len(data_loader_val)),
                np.mean(
                    classification_loss.item() /
                    len(data_loader_val))))
        if epoch > 2:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            torch.save(inter_frame_matcher.state_dict(), model_dir +
                       '/EX+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, epoch_loss))
            torch.save(intra_frame_matcher.state_dict(), model_dir +
                       '/EN+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, epoch_loss))
            torch.save(feature_extract_net.state_dict(), model_dir +
                       '/U-ext+-xstr0-{:.1f}-{:.4f}.pth'.format(epoch, epoch_loss))
