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
from recoloss import sort_feature, sort_labels, run_unet_on_patches
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
parser.add_argument('--out_dir', type=str, required=True,
                    help="Path to the output data directory")
parser.add_argument('--cpu',
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
data_dir_path = args.data_dir
output_dir_path = args.out_dir
feature_extract_net_path = args.model1_dir
mlp_model1_path = args.model2_dir
mlp_model2_path = args.model3_dir
test_index = int(args.test)

# -----------------------------------------------------------------------------
# Dataset Loading
# -----------------------------------------------------------------------------
test_index_str = str(test_index)
image_file_list = os.listdir(data_dir_path + '/mskcc_confocal_s' + test_index_str + '/images/')
image_file_list = [i for i in image_file_list if 'tif' in i]

image_path_dict = {}
for filename in tqdm(image_file_list):
    if 'tif' in filename:
        frame_number = int(filename.split('_')[-1].split('.')[0][1:])
        if frame_number < 273:
            image_path_dict[frame_number] = data_dir_path + '/mskcc_confocal_s' + test_index_str + '/images/' + filename

track_data_df = pd.read_table(data_dir_path + '/mskcc_confocal_s' + test_index_str + '/tracks/tracks.txt')
polar_body_track_data_df = pd.read_table(data_dir_path + '/mskcc_confocal_s' + test_index_str + '/tracks/tracks_polar_bodies.txt')


# -----------------------------------------------------------------------------
# Configure the data loader
# -----------------------------------------------------------------------------

class VDataset(Dataset):

    def __init__(self, image_path_dict):
        self.data = image_path_dict

    def __len__(self):
        return len(self.data) - 5

    def __getitem__(self, index):
        current_index = index
        image_path_1 = self.data[current_index]
        offset = 1
        image_path_2 = self.data[current_index + offset]
        image_path_3 = self.data[current_index + offset + offset]
        image_1 = tifffile.TiffFile(image_path_1).asarray().transpose([1, 2, 0])
        image_2 = tifffile.TiffFile(image_path_2).asarray().transpose([1, 2, 0])
        image_3 = tifffile.TiffFile(image_path_3).asarray().transpose([1, 2, 0])
        image_tensor_1 = torch.from_numpy(image_1.astype(float))
        image_tensor_3 = torch.from_numpy(image_3.astype(float))
        image_tensor_2 = torch.from_numpy(image_2.astype(float))
        return {'image': image_tensor_1, 'image2': image_tensor_2, 'image3': image_tensor_3}


# -----------------------------------------------------------------------------
# Load the model
# -----------------------------------------------------------------------------

batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if args.cpu:
    device = 'cpu'
else:
    device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

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


inference_dataset = VDataset(image_path_dict)
inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0)
inference_progress_bar = tqdm(inference_loader, desc="Iteration")
feature_extract_net.eval()
intra_frame_mathcer.eval()
inter_frame_matcher.eval()


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

def transform(tensor):
    tensor = torch.log1p(tensor)
    return tensor


# -----------------------------------------------------------------------------
# Load and test the final frame to determine the processing scope
# -----------------------------------------------------------------------------


image_frame_269 = tifffile.TiffFile(image_path_dict[269]).asarray().transpose([1, 2, 0]).astype(float)
image_frame_270 = tifffile.TiffFile(image_path_dict[270]).asarray().transpose([1, 2, 0]).astype(float)
labels_tensor = torch.zeros(image_frame_269.shape).unsqueeze(0).to(device)

with torch.no_grad():
    center_mask_1, features, probs, division_map_1, labels, seg_output = run_unet_on_patches(
        feature_extract_net,
        transform(torch.cat([
            torch.from_numpy(image_frame_269).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(image_frame_270).to(device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        ], dim=1)),
        labels_tensor
    )


def compute_crop_box(segmentation_tensor):
    """
    Estimate the bounding box region from predicted segmentation.
    """
    binary_mask = segmentation_tensor > 1

    # z-dimension
    proj_z = list(binary_mask.sum((0, 1)) > 10)
    z_start = proj_z.index(1)
    z_end = len(proj_z) - proj_z[::-1].index(1)

    # x-dimension
    proj_x = list(binary_mask.sum((1, 2)) >= 1)
    x_start = proj_x.index(1)
    x_end = len(proj_x) - proj_x[::-1].index(1)

    # y-dimension
    proj_y = list(binary_mask.sum((0, 2)) >= 1)
    y_start = proj_y.index(1)
    y_end = len(proj_y) - proj_y[::-1].index(1)

    return x_start, x_end, y_start, y_end, z_start, z_end


x_start, x_end, y_start, y_end, z_start, z_end = compute_crop_box(seg_output.cpu())
min_intensity_threshold = image_frame_269[seg_output >= 1].min()

crop_origin = [max(2, x_start - 10), max(2, y_start - 10), 1]
crop_width = max(256, x_end - x_start) + 10
crop_height = max(256, y_end - y_start) + 10
crop_depth = max(image_frame_269.shape[-1] - crop_origin[-1], z_end - z_start- crop_origin[-1])


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
    tensor[tensor < min_intensity_threshold] = min_intensity_threshold
    tensor = tensor - tensor.min()
    tensor = tensor + 1900
    tensor = torch.log1p(tensor)
    return tensor


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

frame_index = 0  # Start from first frame
link_dict = {}
centroid_dict = {}
for step, batch in enumerate(inference_progress_bar):
    # -----------------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------------

    image_tensor_1 = batch["image"].unsqueeze(1)[
                     :, :,
                     crop_origin[0]:crop_origin[0] + crop_width,
                     crop_origin[1]:crop_origin[1] + crop_height,
                     crop_origin[2]:crop_origin[2] + crop_depth
                     ].to(device, dtype=torch.float)

    image_tensor_2 = batch["image2"].unsqueeze(1)[
                     :, :,
                     crop_origin[0]:crop_origin[0] + crop_width,
                     crop_origin[1]:crop_origin[1] + crop_height,
                     crop_origin[2]:crop_origin[2] + crop_depth
                     ].to(device, dtype=torch.float)

    image_tensor_3 = batch["image3"].unsqueeze(1)[
                     :, :,
                     crop_origin[0]:crop_origin[0] + crop_width,
                     crop_origin[1]:crop_origin[1] + crop_height,
                     crop_origin[2]:crop_origin[2] + crop_depth
                     ].to(device, dtype=torch.float)

    # -----------------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------------

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
            coords_frame1, features_frame1, probs_frame1, sizes_frame1, labels_frame1, _ = run_unet_on_patches(
                feature_extract_net,
                preprocess_log1p(torch.cat([image_tensor_1, image_tensor_2], dim=1)),
                image_tensor_1[:, 0] * 0
            )
        else:
            coords_frame1 = prev_frame_coords.clone()
            features_frame1 = prev_frame_features.clone()
            probs_frame1 = prev_frame_probs.clone()
            sizes_frame1 = prev_frame_sizes.clone()
            labels_frame1 = labels_tensor[
                coords_frame1[:, 0], coords_frame1[:, 1], coords_frame1[:, 2], coords_frame1[:, 3]]

        coords_frame2, features_frame2, probs_frame2, sizes_frame2, labels_frame2, _ = run_unet_on_patches(
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
    coords_frame1[:, 2] *= 5  # scale z
    coords_frame2[:, 2] *= 5  # scale z

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
        if step > 0:
            intra_frame_group = intra_frame2_group_main  # use previous frame group
            frame_feature_query_indices = curr_frame_nn_indices.clone()
        else:
            m, n = query_frame_positions.shape[0], query_frame_positions.shape[0]
            query_pos_proj = query_frame_positions.clone()
            query_pos_proj[:, 2] = query_pos_proj[:, 2] // 5

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
            valid_queries = []
            intra_frame_group = {}
            intra_cell_clusters = {}
            for i in range(query_frame_positions.shape[0]):
                if i not in matched_indices:
                    current_group = [i]
                    if strong_sim_count[i] > 0:
                        for j in range(4):
                            if high_conf_mask[i, j] or (
                                    moderate_conf_mask[i, j] and positions_k1_2[i, j] < max(5, 0.6 * sizes_frame1[i])):
                                matched_indices.append(dist_indices[i, j].item())
                                current_group.append(dist_indices[i, j].item())
                    intra_frame_group[i] = current_group
                    valid_queries.append(i)
        # -----------------------------------------------------------------------------
        # frame 2

        m, n = curr_frame_positions.shape[0], curr_frame_positions.shape[0]
        curr_pos_proj = curr_frame_positions.clone()
        curr_pos_proj[:, 2] = curr_pos_proj[:, 2] // 5

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
                current_centroids[anchor_id] = frame1_centroids[anchor_id] + crop_origin

        link_dict[step] = current_links  
        centroid_dict[step] = current_centroids  
        prev_frame_feature_map = frame_cell_to_group.copy()

# -----------------------------------------------------------------------------
# Integrate data, remove excessively short paths, and output trajectories
# -----------------------------------------------------------------------------
cell_instance_dict = {}  
trajectory_dict = {}      
trajectory_timepoints = {}  
cell_id_counter = 1
track_id_counter = 1

query_id_map = {} 
parent_id_map = {}  # pid
descendant_id_map = {}  # did

initial_centroids = centroid_dict[0].copy()
temp_map = {}


for anchor_id in link_dict[0]:
    if isinstance(link_dict[0][anchor_id], list) and len(link_dict[0][anchor_id]) > 0:
        cell_instance_dict[cell_id_counter] = [
            cell_id_counter, track_id_counter, 0,
            initial_centroids[anchor_id][0].item(),
            initial_centroids[anchor_id][1].item(),
            initial_centroids[anchor_id][2].item(),
            -1  # parentid
        ]
        trajectory_dict[track_id_counter] = trajectory_dict.get(track_id_counter, []) + [cell_id_counter]
        trajectory_timepoints[track_id_counter] = trajectory_timepoints.get(track_id_counter, []) + [0]

        for match in link_dict[0][anchor_id]:
            temp_map[match] = cell_id_counter
        cell_id_counter += 1
        track_id_counter += 1


for t in tqdm(range(1, 267)):
    next_temp_map = {}
    current_centroids = centroid_dict[t]
    for anchor_id in link_dict[t]:
        prev_instance_id = temp_map.get(anchor_id, -1)
        if prev_instance_id in cell_instance_dict:
            traj_id = cell_instance_dict[prev_instance_id][1]
        else:
            track_id_counter += 1
            traj_id = track_id_counter

        cell_instance_dict[cell_id_counter] = [
            cell_id_counter,
            traj_id,
            t,
            current_centroids[anchor_id][0].item(),
            current_centroids[anchor_id][1].item(),
            current_centroids[anchor_id][2].item(),
            prev_instance_id  # parentid
        ]
        trajectory_dict[traj_id] = trajectory_dict.get(traj_id, []) + [cell_id_counter]
        trajectory_timepoints[traj_id] = trajectory_timepoints.get(traj_id, []) + [t]
        for match in link_dict[t][anchor_id]:
            next_temp_map[match] = cell_id_counter
        cell_id_counter += 1
    temp_map = next_temp_map.copy()


parent_mapping = {}
for cell_id in cell_instance_dict:
    parent_mapping[cell_id] = cell_instance_dict[cell_id][-1]

reverse_mapping = {}
for cid in parent_mapping:
    reverse_mapping[parent_mapping[cid]] = reverse_mapping.get(parent_mapping[cid], []) + [cid]

short_tracks = []
for traj_id in trajectory_timepoints:
    if len(trajectory_timepoints[traj_id]) < 3 and max(trajectory_timepoints[traj_id]) < 260 \
       and max(trajectory_dict[traj_id]) not in reverse_mapping and min(trajectory_timepoints[traj_id]) > 0:
        short_tracks.append(traj_id)

for cid in list(cell_instance_dict.keys()):
    if cell_instance_dict[cid][1] in short_tracks:
        cell_instance_dict.pop(cid)


tracking_output_df = pd.DataFrame()
cell_ids, track_ids, time_frames = [], [], []
x_coords, y_coords, z_coords, parent_ids = [], [], [], []

for cid in cell_instance_dict:
    instance = cell_instance_dict[cid]
    cell_ids.append(instance[0])
    track_ids.append(instance[1])
    time_frames.append(instance[2])
    x_coords.append(instance[3])
    y_coords.append(instance[4])
    z_coords.append(instance[5])
    parent_ids.append(instance[6])

tracking_output_df['cellid'] = cell_ids
tracking_output_df['trackid'] = track_ids
tracking_output_df['t'] = time_frames
tracking_output_df['x'] = x_coords
tracking_output_df['y'] = y_coords
tracking_output_df['z'] = z_coords
tracking_output_df['parentid'] = parent_ids

tracking_output_df.to_csv('track-result.csv', index=False)
