import os
import re
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
from recoloss import sort_feature, sort_labels, run_unet_on_patches_infer
from recoloss import kflb as search_mask
from recoloss import ud as z_enlarge
from unetext3Dn_con7 import UNet3D, EXNet

# Ensure KMP duplicates do not error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Argument parser setup
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inference script for CELLECT")

parser.add_argument('--data_dir', type=str, required=True,
                    help="Path to the input image directory")
parser.add_argument('--out_dir', type=str, required=True,
                    help="Path to the output result directory")
parser.add_argument('--cpu', action='store_true',
                    help="Use CPU instead of GPU if this flag is set.")

parser.add_argument('--model1_dir', type=str, required=True,
                    help="Path to the Unet model")
parser.add_argument('--model2_dir', type=str, required=True,
                    help="Path to MLP model 1")
parser.add_argument('--model3_dir', type=str, required=True,
                    help="Path to MLP model 2")


# -----------------------------------------------------------------------------
# Additional configurable parameters (with defaults)
# -----------------------------------------------------------------------------
parser.add_argument('--zratio', type=float, default=5.0,
                    help="Z-axis to XY resolution ratio (default: 5)")
parser.add_argument('--suo', type=int, default=1,
                    help="XY pooling (scaling) factor (default: 1)")
parser.add_argument('--high', type=int, default=65535,
                    help="Upper bound for image intensity (default: 65535)")
parser.add_argument('--low', type=int, default=0,
                    help="Lower bound for image intensity (default: 0)")
parser.add_argument('--thresh0', type=int, default=0,
                    help="Threshold for zeroing low intensity values (default: 0)")
parser.add_argument('--div', type=int, default=0,
                    help="Allow division/splitting (1=enable, 0=disable, default: 0)")
parser.add_argument('--enhance', type=float, default=1.0,
                    help="Enhancement factor to increase/decrease detection sensitivity (default: 1.0)")

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
args = parser.parse_args()

# Assign to variables
data_dir_path = args.data_dir
output_dir_path = args.out_dir
feature_extract_net_path = args.model1_dir
mlp_model1_path = args.model2_dir
mlp_model2_path = args.model3_dir
# Additional params
zratio = int(args.zratio)
suo = args.suo
high = args.high
low = args.low
thresh0 = args.thresh0
div = args.div
enhance = args.enhance
# data_dir_path='C:/Users/z/Desktop/CELLECT/microglia_stroke_data/'
# #data_dir_path='../extradata/mskcc-confocal/mskcc_confocal_s1/images'
# output_dir_path='./'
# feature_extract_net_path='./model/U-ext+-x3rd-149.0-4.6540.pth'
# mlp_model1_path='./model/EX+-x3rd-149.0-4.6540.pth'
# mlp_model2_path='./model/EN+-x3rd-149.0-4.6540.pth'
# mp1='./model/U-ext+-x3rdstr0-149.0-3.4599.pth'
# mp2='./model/EX+-x3rdstr0-149.0-3.4599.pth'
# mp3='./model/EN+-x3rdstr0-149.0-3.4599.pth'
# feature_extract_net_path='./model/U-ext+-x3rdstr0-149.0-3.4599.pth'
# mlp_model1_path='./model/EX+-x3rdstr0-149.0-3.4599.pth'
# mlp_model2_path='./model/EN+-x3rdstr0-149.0-3.4599.pth'
# test_index=2
# zratio=5
# suo=1
# high=65535
# low=0
# thresh0=0
# div=0
# enhance=1
output_dir_path1=output_dir_path+'/results/'
os.makedirs(output_dir_path1, exist_ok=True)

# -----------------------------------------------------------------------------
# Load the model
# -----------------------------------------------------------------------------

batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if 0:#args.cpu:
    device = 'cpu'
else:
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
try:
    feature_extract_net = UNet3D(2, 6)
    inter_frame_matcher = EXNet(64, 8)
    intra_frame_mathcer = EXNet(64, 6)
    
    feature_extract_net.to(device)
    inter_frame_matcher.to(device)
    intra_frame_mathcer.to(device)
    
    if device == 'cpu':
        feature_extract_net.load_state_dict(torch.load(feature_extract_net_path, map_location=torch.device('cpu')))
        inter_frame_matcher.load_state_dict(torch.load(mlp_model1_path, map_location=torch.device('cpu')))
        intra_frame_mathcer.load_state_dict(torch.load(mlp_model2_path, map_location=torch.device('cpu')))
    else:
        feature_extract_net.load_state_dict(torch.load(feature_extract_net_path))
        inter_frame_matcher.load_state_dict(torch.load(mlp_model1_path))
        intra_frame_mathcer.load_state_dict(torch.load(mlp_model2_path))
except:
    from unetext3Dn_con7s import UNet3D, EXNet
    feature_extract_net = UNet3D(2, 6)
    inter_frame_matcher = EXNet(64, 8)
    intra_frame_mathcer = EXNet(64, 6)
    
    feature_extract_net.to(device)
    inter_frame_matcher.to(device)
    intra_frame_mathcer.to(device)
    
    if device == 'cpu':
        feature_extract_net.load_state_dict(torch.load(feature_extract_net_path, map_location=torch.device('cpu')))
        inter_frame_matcher.load_state_dict(torch.load(mlp_model1_path, map_location=torch.device('cpu')))
        intra_frame_mathcer.load_state_dict(torch.load(mlp_model2_path, map_location=torch.device('cpu')))
    else:
        feature_extract_net.load_state_dict(torch.load(feature_extract_net_path))
        inter_frame_matcher.load_state_dict(torch.load(mlp_model1_path))
        intra_frame_mathcer.load_state_dict(torch.load(mlp_model2_path))
print('    Total params: %.2fM' % (sum(p.numel() for p in feature_extract_net.parameters()) / 1000000.0))

feature_extract_net.eval()
intra_frame_mathcer.eval()
inter_frame_matcher.eval()



# -----------------------------------------------------------------------------
# Update preprocessing
# -----------------------------------------------------------------------------

def preprocess_log1p(tensor):
    """
    Preprocess function:
    - Applies minimum intensity threshold (based on segmentation result).
    - Rescales intensities to start near 2000.
    - Applies log1p for compression.

    These adjustments help match the training distribution.
    """
    #tensor[tensor < min_intensity_threshold] = min_intensity_threshold
    tmin = tensor[tensor>0].min()
    tensor[tensor < tmin] = tmin
    tensor = tensor * enhance
    tensor = tensor + 1900
    tensor = torch.log1p(tensor)
    return tensor
def t3d(x):
    x[x > high] = high
    x[x < low] = low
    x[x <thresh0] = 0
    x1=F.max_pool3d(torch.from_numpy(x.astype(float)).squeeze().unsqueeze(0),(suo,suo,1)).numpy()[0]
    return x1
def tim(x):
    #load tif
    #return shape [X, Y, Z]
    x = tifffile.TiffFile(x).asarray().transpose([1,2,0])
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
def extract_frame_number(filename):
    """
    提取文件名中的帧编号（整数），支持多种格式如 _1, t003, frame5 等
    """
    matches = re.findall(r'(?:_t?|frame)?(\d+)', filename.lower())
    return int(matches[-1]) if matches else None

def build_frame_dict(folder_path):
    frame_path_dict = {}
    
    # 遍历文件夹
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue  # 跳过非tif文件
        num = extract_frame_number(fname)
        if num is None:
            continue  # 跳过无数字文件
        full_path = os.path.join(folder_path, fname)
        frame_path_dict[num] = full_path  # 数字为key，路径为值

    sorted_keys = sorted(frame_path_dict.keys())  # 按帧号排序
    return frame_path_dict, sorted_keys

path_dict, order_list = build_frame_dict(data_dir_path)

frame_index = 0  # Start from first frame
link_dict = {}
centroid_dict = {}
nid=0
kid={}
kidn=0
for step in tqdm(range(len(order_list)-2)):
    # -----------------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------------


    if step==0:
        i1,i2,i3=t3d(tim(path_dict[order_list[step]])),t3d(tim(path_dict[order_list[step+1]])),t3d(tim(path_dict[order_list[step+2]]))#t3d(tim(Image.open(Dfile[Dl[st]]))),t3d(tim(Image.open(Dfile[Dl[st+1]]))),t3d(tim(Image.open(Dfile[Dl[st+2]])))

    else:
        i1=i2.copy()
        i2=i3.copy()
        i3=t3d(tim(path_dict[order_list[step+2]]))
        kid=kid2
    image_tensor_1 = torch.from_numpy(i1).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    
    image_tensor_2 = torch.from_numpy(i2).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    
    image_tensor_3 = torch.from_numpy(i3).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)   
    # -----------------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------------
    kid2={}
    with torch.no_grad():
        if step == 0:
            """
            Sliding-window inference over the full volume via patches.
        
            Parameters
            ----------
            feature_extract_net : nn.Module
                Pretrained 3D U‐Net model.
                volume_tensor :  shape (1, 2, H, W, Z), 2 frames input tensor (batch size B).
                label_tensor : Full volume ground-truth labels.
        
            Returns
            -------
            coords_all, features_all, probs_all, sizes_all, labels_all, seg_mask : tuple
                - coords_all: Tensor[M,4] all coordinates
                - features_all: Tensor[M,C] all feature vectors
                - probs_all: Tensor[M,2] all probabilities
                - sizes_all: Tensor[M] all sizes
                - labels_all: Tensor[M] all labels
                - seg_mask: Tensor[H,W,Z] aggregated UNet prediction
            """            
            coords_frame1, features_frame1, probs_frame1, sizes_frame1, labels_frame1, _ = run_unet_on_patches_infer(
                feature_extract_net,
                preprocess_log1p(torch.cat([image_tensor_1, image_tensor_2], dim=1)),
                image_tensor_1[:, 0] * 0
            )
        else:
            coords_frame1 = prev_frame_coords.clone()
            features_frame1 = prev_frame_features.clone()
            probs_frame1 = prev_frame_probs.clone()
            sizes_frame1 = prev_frame_sizes.clone()
            labels_frame1 = sizes_frame1

        coords_frame2, features_frame2, probs_frame2, sizes_frame2, labels_frame2, _ = run_unet_on_patches_infer(
            feature_extract_net,
            preprocess_log1p(torch.cat([image_tensor_2, image_tensor_3], dim=1)),
            image_tensor_2[:, 0] * 0
        )

    prev_frame_coords = coords_frame2.clone()
    prev_frame_features = features_frame2.clone()
    prev_frame_probs = probs_frame2.clone()
    prev_frame_sizes = sizes_frame2.clone()

    # Project z-coordinate to match physical resolution
    coords_frame1 = coords_frame1[:, 1:]
    coords_frame2 = coords_frame2[:, 1:]
    coords_frame1[:, 2] *= zratio  # scale z
    coords_frame2[:, 2] *= zratio  # scale z

    # Prepare frame 1 and frame 2 features/coordinates/labels for intra-frame exclusion
    query_frame_features = features_frame1
    query_frame_labels = labels_frame1
    query_frame_positions = coords_frame1
    curr_frame_features = features_frame2
    curr_frame_labels = labels_frame2
    curr_frame_positions = coords_frame2

    if query_frame_features.shape[0] > 2 and curr_frame_features.shape[0] > 2:
        # -------------------------------------------------------------------------
        # Use MLP to find points in the same frame that belong to the same cell
        # -------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # frame 1
        try: tnid=tnid2
        except:tnid={}
        tnid2={}
        if step > 0:
            intra_frame_group = intra_frame2_group_main  # use previous frame group
            frame_feature_query_indices = curr_frame_nn_indices.clone()
        else:
            m, n = query_frame_positions.shape[0], query_frame_positions.shape[0]
            query_pos_proj = query_frame_positions.clone()
            query_pos_proj[:, 2] = query_pos_proj[:, 2] // zratio

            # Euclidean distance for intra-frame points
            dist_matrix = torch.pow(query_pos_proj.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(query_pos_proj.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_matrix.addmm_(1, -2, query_pos_proj.float(), query_pos_proj.float().t())

            dist_values, dist_indices = dist_matrix.topk(min(6, query_pos_proj.shape[0]), largest=False)

            if dist_indices.shape[1] < 6:
                while dist_indices.shape[1] < 6:
                    dist_indices = torch.cat([dist_indices, dist_indices[:, -1].unsqueeze(-1)], dim=1)
            dist_indices = dist_indices[:, 1:] 

            features_k1, positions_k1, size_k1, division_k1 = sort_feature(
                query_frame_features, query_frame_positions, query_frame_positions,
                dist_indices, sizes_frame1, sizes_frame1, probs_frame1, probs_frame1, n=5
            )
            positions_k1_2 = torch.sqrt((positions_k1 * positions_k1).sum(-1))

            with torch.no_grad():
                similarity_scores = F.sigmoid(
                    intra_frame_mathcer(query_frame_features.unsqueeze(1), features_k1, positions_k1, 
                                        size_k1, division_k1))
             

            label_neighbors = []
            for j in range(5):
                label_neighbors.append((query_frame_labels[dist_indices[:, j]]).unsqueeze(1))
            label_neighbors = torch.cat(label_neighbors, 1)

            strong_sim_count = (similarity_scores[:, :-1] > 0.5).sum(1)
            high_conf_mask = (similarity_scores > 0.9999)
            moderate_conf_mask = (similarity_scores > 0.6)

            matched_indices = []
            valid_queries1 = []
            intra_frame_group = {}
            intra_cell_clusters = {}
            for i in range(query_frame_positions.shape[0]):
                if step==0:
                    tnid[i]=nid+i
                if i not in matched_indices:
                    current_group = [i]
                    if strong_sim_count[i] > 0:
                        for j in range(4):
                            if high_conf_mask[i, j] or (
                                    moderate_conf_mask[i, j] and positions_k1_2[i, j] < max(5, 0.6 * sizes_frame1[i])):
                                matched_indices.append(dist_indices[i, j].item())
                                current_group.append(dist_indices[i, j].item())
                    intra_frame_group[i] = intra_frame_group.get(i, []) + current_group
                    valid_queries1.append(i)

            ndx={}
            nl=[]
            for i in intra_frame_group:
                if i not in nl:
                    nll=list(np.unique(intra_frame_group[i]))
                    ndx[i]=nll
                    nl+=nll

            intra_frame_group=ndx
        # -----------------------------------------------------------------------------
        # frame 2
        tnd=nid+len(tnid)
        m, n = curr_frame_positions.shape[0], curr_frame_positions.shape[0]
        curr_pos_proj = curr_frame_positions.clone()
        curr_pos_proj[:, 2] = curr_pos_proj[:, 2] // zratio

        dist_matrix = torch.pow(curr_pos_proj.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(curr_pos_proj.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_matrix.addmm_(1, -2, curr_pos_proj.float(), curr_pos_proj.float().t())

        _, dist_indices = dist_matrix.topk(min(6, curr_pos_proj.shape[0]), largest=False)

        if dist_indices.shape[1] < 6:
            while dist_indices.shape[1] < 6:
                dist_indices = torch.cat([dist_indices, dist_indices[:, -1].unsqueeze(-1)], dim=1)
        dist_indices = dist_indices[:, 1:]
        curr_frame_nn_indices = dist_indices.clone()

        features_k1, positions_k1, size_k1, division_k1 = sort_feature(
            curr_frame_features, curr_frame_positions, curr_frame_positions,
            dist_indices, sizes_frame2, sizes_frame2, probs_frame2, probs_frame2, n=5
        )
        positions_k1_2 = torch.sqrt((positions_k1 * positions_k1).sum(-1))

        with torch.no_grad():
            similarity_scores = F.sigmoid(
                intra_frame_mathcer(curr_frame_features.unsqueeze(1), features_k1, positions_k1, 
                                    size_k1, division_k1))

        label_neighbors = []
        for j in range(5):
            label_neighbors.append((curr_frame_labels[dist_indices[:, j]]).unsqueeze(1))
        label_neighbors = torch.cat(label_neighbors, 1)

        strong_sim_count = (similarity_scores[:, :-1] > 0.5).sum(1)
        high_conf_mask = (similarity_scores > 0.9999)
        moderate_conf_mask = (similarity_scores > 0.6)

        matched_indices = []
        valid_queries = []
        intra_frame2_group = {}
        intra_frame2_group_main = {}
        intra_frame2_mapping = {}

        for i in range(curr_frame_positions.shape[0]):
            current_group = [i]
            intra_frame2_mapping[i] = i
            if strong_sim_count[i] > 0:
                for j in range(4):
                    if high_conf_mask[i, j] or (
                            moderate_conf_mask[i, j] and positions_k1_2[i, j] < max(5, 0.6 * sizes_frame2[i])):
                        current_group.append(dist_indices[i, j].item())
                        intra_frame2_mapping[dist_indices[i, j].item()] = i
                        matched_indices.append(dist_indices[i, j].item())
            intra_frame2_group[i] = current_group
            tnid2[i]=tnd+1+i
            if i not in matched_indices:
                intra_frame2_group_main[i] = current_group

        # Reverse map from merged to anchor
        frame_cell_to_group = {}
        for anchor_id in intra_frame2_group_main:
            for member_id in intra_frame2_group_main[anchor_id]:
                frame_cell_to_group[member_id] = anchor_id
            valid_queries.append(anchor_id)


        # -----------------------------------------------------------------------------
        # Link cells across different frames
        # -----------------------------------------------------------------------------
        m, n = query_frame_features.shape[0], curr_frame_features.shape[0]
        dist_matrix = torch.pow(query_frame_positions.float(), 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(curr_frame_positions.float(), 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_matrix.addmm_(1, -2, query_frame_positions.float(), curr_frame_positions.float().t())

        _, dist_indices = dist_matrix.topk(min(5, query_frame_positions.shape[0], curr_frame_positions.shape[0]),
                                          largest=False)

        if dist_indices.shape[1] < 5:
            while dist_indices.shape[1] < 5:
                dist_indices = torch.cat([dist_indices, dist_indices[:, -1].unsqueeze(-1)], dim=1)

        features_k1, positions_k1, size_k1, division_k1 = sort_feature(
            curr_frame_features, curr_frame_positions, query_frame_positions,
            dist_indices, sizes_frame2, sizes_frame1, probs_frame2, probs_frame1, n=5
        )
        positions_k1_2 = torch.sqrt((positions_k1 * positions_k1).sum(-1))

        with torch.no_grad():
            score_logits = inter_frame_matcher(query_frame_features.unsqueeze(1), features_k1, positions_k1, size_k1, division_k1 )
            soft_scores = F.softmax(score_logits[:, -2:], dim=-1)
            similarity_scores = F.sigmoid(score_logits[:, :-2])

        label_neighbors = []
        for j in range(5):
            label_neighbors.append((curr_frame_labels[dist_indices[:, j]]).unsqueeze(1))
        label_neighbors = torch.cat(label_neighbors, 1)

        cross_frame_link_dict = {}
        cross_frame_score_dict = {}
        cross_frame_distance_dict = {}
        cross_frame_quality_dict = {}
        cross_frame_size_dict = {}
        frame1_centroids = {}

        for query_idx in intra_frame_group:
            target_ids = intra_frame_group[query_idx]
            if step > 0:
                anchor_id = prev_frame_feature_map[min(target_ids)]
            else:
                anchor_id = min(target_ids)

            group_centroid = torch.cat([query_frame_positions[i].unsqueeze(0) for i in target_ids], 0).float().mean(0).cpu().numpy()
            frame1_centroids[anchor_id] = group_centroid

            candidate_ids = []
            candidate_scores = []
            candidate_dists = []
            candidate_group_ids = []

            size_anchor = max([sizes_frame1[i].item() for i in target_ids])
            confidence_anchor = soft_scores[query_idx][1].item()

            for rank in range(5):
                candidate_idx = dist_indices[query_idx, rank].item()
                if similarity_scores[query_idx].argmax(0) < 5:
                    if rank == 0 or candidate_idx not in candidate_group_ids:
                        candidate_ids.append(candidate_idx)
                        candidate_scores.append(similarity_scores[query_idx, rank].item())
                        candidate_dists.append(positions_k1_2[query_idx, rank].item())
                        candidate_group_ids += intra_frame2_group.get(candidate_idx, [])

            if len(candidate_ids) > 0:
                dedup_scores = {}
                dedup_dists = {}
                for i in range(len(candidate_ids)):
                    cid = candidate_ids[i]
                    dedup_scores[cid] = max(dedup_scores.get(cid, 0), candidate_scores[i])
                    dedup_dists[cid] = candidate_dists[i]

                filtered_ids = list(dedup_scores.keys())
                filtered_scores = [dedup_scores[cid] for cid in filtered_ids]
                filtered_dists = [dedup_dists[cid] for cid in filtered_ids]

                intra_frame2_group[100000] = []
                filtered_ids += cross_frame_link_dict.get(anchor_id, [100000])
                filtered_scores += cross_frame_score_dict.get(anchor_id, [0])
                filtered_dists += cross_frame_distance_dict.get(anchor_id, [0])


                top_k = torch.from_numpy(np.array(filtered_scores)).topk(min(5, len(filtered_scores)))[1]
                final_ids = []
                final_scores = []
                final_dists = []
                seen = set()

                for k in top_k:
                    cid = filtered_ids[k]
                    if cid == 100000:
                        break
                    if cid not in seen and filtered_dists[k] < max(size_anchor, 5) * 6:
                        seen.update(intra_frame2_group[cid])
                        final_ids.append(cid)
                        final_scores.append(filtered_scores[k])
                        final_dists.append(filtered_dists[k])

                cross_frame_link_dict[anchor_id] = final_ids[:2]
                cross_frame_score_dict[anchor_id] = final_scores[:2]
                cross_frame_distance_dict[anchor_id] = final_dists[:2]
                cross_frame_quality_dict[anchor_id] = confidence_anchor
                cross_frame_size_dict[anchor_id] = size_anchor


        final_cross_link_dict = {}
        if div:
            for anchor_id in cross_frame_link_dict:
                links = cross_frame_link_dict[anchor_id]
                final_cross_link_dict[anchor_id] = [
                    [frame_cell_to_group.get(target_id, -1), cross_frame_score_dict[anchor_id][i]]
                    for i, target_id in enumerate(links)
                    if target_id != 100000 and (
                        (i == 0 and cross_frame_score_dict[anchor_id][0] > 0.1) or
                        (i > 0 and cross_frame_quality_dict[anchor_id] > 0.8 or (
                            cross_frame_score_dict[anchor_id][-1] > 0.5 and
                            cross_frame_quality_dict[anchor_id] > 0.5 and
                            abs(cross_frame_distance_dict[anchor_id][-1] - cross_frame_distance_dict[anchor_id][0])
                            < 1.2 * cross_frame_size_dict[anchor_id]
                        ))
                    )
                ]
        else:
                for anchor_id in cross_frame_link_dict:
                    links = cross_frame_link_dict[anchor_id]
                    final_cross_link_dict[anchor_id] = [
                        [frame_cell_to_group.get(target_id, -1), cross_frame_score_dict[anchor_id][i]]
                        for i, target_id in enumerate(links)
                        if target_id != 100000 and (
                            (i == 0 and cross_frame_score_dict[anchor_id][0] > 0.1)
                        )
                    ]
        resolved_links = {}
        for anchor_id in final_cross_link_dict:
            for target_info in final_cross_link_dict[anchor_id]:
                target_anchor = target_info[0]
                score = target_info[1]
                if resolved_links.get(target_anchor, 0) < score:
                    resolved_links[target_anchor] = score

        for anchor_id in final_cross_link_dict:
            final_cross_link_dict[anchor_id] = [
                target_info for target_info in final_cross_link_dict[anchor_id]
                if resolved_links.get(target_info[0], 0) <= target_info[1]
            ]

        filtered_cross_link_dict = final_cross_link_dict.copy()
        
        
        
        # -----------------------------------------------------------------------------
        # Reconnect broken trajectories if feature similarity supports it
        # -----------------------------------------------------------------------------
        if step > 0:
           prev_frame_id_values = list(prev_frame_feature_map.values())
           for anchor_id in final_cross_link_dict:
                if len(final_cross_link_dict[anchor_id]) > 0:
                    if anchor_id not in previous_links:
                        for dim in range(3):
                            fallback_query = query_frame_positions[anchor_id][dim].item()
                            if fallback_query in prev_frame_id_values:
                                prev_anchor = prev_frame_feature_map[fallback_query]
                                if prev_anchor in previous_links and prev_anchor in frame1_centroids:
                                    # merge current to previous
                                    filtered_cross_link_dict[prev_anchor] = (
                                        filtered_cross_link_dict.get(prev_anchor, []) +
                                        final_cross_link_dict[anchor_id]
                                    )
                                    filtered_cross_link_dict.pop(anchor_id)
                                    break 
        # -----------------------------------------------------------------------------
        # Log matched links
        # -----------------------------------------------------------------------------
        current_links = {}
        current_centroids = {}
        previous_links = []

        for anchor_id in filtered_cross_link_dict:
            if anchor_id in frame1_centroids:
                current_links[anchor_id] = [target[0] for target in filtered_cross_link_dict[anchor_id]]
                previous_links += current_links[anchor_id]
                current_centroids[anchor_id] = frame1_centroids[anchor_id]

        link_dict[step] = current_links  
        centroid_dict[step] = current_centroids  
        prev_frame_feature_map = frame_cell_to_group.copy()
        for i in link_dict:
            if i < step-1000:
                link_dict.pop(i)
                centroid_dict.pop(i)
    
        zr2 = {}
        for i in link_dict[step]:
            for j in link_dict[step][i]:
                zr2[j] = i
        dux=np.unique([zr2[i] for i in zr2])
        if step==0 and dux.shape[0]>0:
            for i in dux:
                kid[i]=i
            kidn=max(dux)+1
            
            
        zd=pd.DataFrame()
        l1,l2,l3,l4,l5,l6,l7,l8,l9=[],[],[],[],[],[],[],[],[]
        lf=[]
        l11=[]
        
        l10=[]
        for i in valid_queries:
            x,y,z=curr_frame_positions[i]
            x=x*suo
            y=y*suo
            l1.append(tnid2[i])
            l2.append(x.item())
            l3.append(y.item())
            l4.append(z.item())
            l11.append(sizes_frame2[i].item()*suo)
            j=zr2.get(i,-1)
            if j==-1:
                l5.append(-1)
                l6.append(-1)
                l7.append(-1)
                l8.append(-1)
                l10.append(kidn)
                kid2[i]=kidn
                kidn+=1
            else:
                px,py,pz=query_frame_positions[j]
                px=px*suo
                py=py*suo
                v=np.sqrt(pow(px.item()-x.item(),2)+pow(y.item()-py.item(),2))
               
                if v<100:
                    l5.append(tnid[j])

                    l6.append(x.item())
                    l7.append(y.item())
                    l8.append(z.item())
                    if j in kid:
                        l10.append(kid[j])
                        kid2[i]=kid[j]
                    else:
                        l10.append(kidn)
                        kid2[i]=kidn
                        kidn+=1        
                else:
                    l5.append(-1)
                    l6.append(-1)
                    l7.append(-1)
                    l8.append(-1)
                    l10.append(kidn)
                    kid2[i]=kidn
                    kidn+=1    
                
            l9.append(step+1)
       
        zd['t']=np.array(l9)
        zd['cellid']=np.array(l1)
        zd['x']=np.array(l2)
        zd['y']=np.array(l3)
        zd['z']=np.array(l4)
        zd['size']=np.array(l11)
        zd['parentid']=np.array(l5)
        zd['px']=np.array(l6)
        zd['py']=np.array(l7)
        zd['pz']=np.array(l8)
        zd['trackid']=np.array(l10)
        if step==0:
            track=zd
        else:
            track=pd.concat([track,zd])
        if step%100==0:
            print('detected:',zd.shape[0])
            track.to_csv(output_dir_path1+str(step)+'.csv',index=False)

        nid+=len(tnid)+len(tnid2)
     
cc=track.groupby('trackid')['t'].nunique().rename('countq').reset_index()
track=track.merge(cc, on='trackid')    
track=track.loc[track.countq>3] 
track.to_csv(output_dir_path+'track.csv',index=False) 