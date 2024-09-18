import os
import pathlib
import torch

home_directory = os.path.expanduser('~')

import numpy as np
import pandas as pd
from utils.metrics import LPIPSMeter, PSNRMeter, SSIM
from utils.general import load_arguments_from_json, object_index_to_object_name, object_name_to_object_index
from mast3r_feature_extractor import MasterFeaturesExtractor

from openpyxl import Workbook
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
import mast3r.utils.path_to_dust3r
from dust3r.utils.image import load_images
import re



try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity is not available under Windows, use
    # os.cpu_count instead (which may not return the *available* number
    # of CPUs).
    num_cpus = os.cpu_count()
num_workers = min(num_cpus, 8) if num_cpus is not None else 0
batch_size = 60


scorers = {
            "psnr": PSNRMeter(size=256),
            # "lpips": LPIPSMeter(device="cuda:0", size=256, net="vgg"),
            # "ssim": SSIM(size=256)
        }

extractor = MasterFeaturesExtractor(size=256, device=device, batch_size=batch_size, num_workers=num_workers)

csv_columns = (
        ["obj", "psnr", "lpips", "ssim"]
        )



# def process_scores_in_batches(scorers, ref_paths, novel_paths, batch_size=16):
#     num_samples = len(ref_paths)
#     temp_scores = {key: [] for key in scorers.keys()}
    
#     for start_idx in range(0, num_samples, batch_size):
#         end_idx = min(start_idx + batch_size, num_samples)
        
#         ref_batch = ref_paths[start_idx:end_idx]
#         novel_batch = novel_paths[start_idx:end_idx]
        
#         for key, scorer in scorers.items():
#             # Score the current batch
#             batch_score = scorer.score_gt(ref_batch, novel_batch)[1]
#             temp_scores[key].extend(batch_score)
    
#     # Convert lists to numpy arrays if necessary
#     temp_scores = {key: np.array(scores) for key, scores in temp_scores.items()}
    
#     return temp_scores


def extract_features_from_exp(exp_root_folder, mast3r_args):
    args = load_arguments_from_json(os.path.join(exp_root_folder, "arguments.json"))
    mast3r_flag = mast3r_args[0]
    mast3r_mode = mast3r_args[1]

    # all source images cache
    all_source_paths = sorted(
        [
            str(file)
            for file in pathlib.Path(args.data_path).rglob("*.png")
            if re.fullmatch(r'\d+', file.stem)  # Include only files where the name is just a number
        ]
    )
    all_img_source =  load_images(all_source_paths, size=extractor.size)
    #

    # all novel_paths and corresponding ref_paths, source_paths, and list of (object_num, source_view_indexm target_view_index)
    delta_azimuthes_in_degrees_str = [d for d in os.listdir(exp_root_folder) if os.path.isdir(os.path.join(exp_root_folder, d))]
    delta_azimuthes_in_degrees = sorted(delta_azimuthes_in_degrees_str, key=lambda x: float(x.replace('delta', '')))
    data_rows = []
    for delta_azimuth_in_degrees in delta_azimuthes_in_degrees:
        delta_azimuth_in_degrees_folder = os.path.join(exp_root_folder, delta_azimuth_in_degrees)
        novel_paths = sorted(
                [
                    str(file)
                    for file in pathlib.Path(delta_azimuth_in_degrees_folder).rglob("*.png")
                ]
            )
        ref_paths = [ # the gt image of each prediction
                    f"{args.data_path}/{s.split('/')[-2]}/{int(s.split('/')[-3].split('to')[1]):03}.png"
                    for s in novel_paths
                   ]
        source_paths = [ # the source view of each prediction
                    f"{args.data_path}/{s.split('/')[-2]}/{int(s.split('/')[-3].split('to')[0]):03}.png"
                    for s in novel_paths
                   ]
        object_num__source_view_index__target_view_index = [(object_name_to_object_index[s.split('/')[-2]], int(s.split('/')[-3].split('to')[0]), int(s.split('/')[-3].split('to')[1])) for s in novel_paths]
        
        # if extracting features with mast3r, make pairs, extract features, and build a dataframe to store all info
        if mast3r_flag:
            img_source = [all_img_source[obj_idx * 16 + view_idx] for obj_idx, view_idx, _ in object_num__source_view_index__target_view_index]
            source_novel_joint_features = extractor.extract_features_single_source_multiple_targets(img_source, novel_paths, mast3r_mode)
            # temp_scores = process_scores_in_batches(scorers, ref_paths, novel_paths, batch_size=len(ref_paths))
            temp_scores = {key: scorer.score_gt(ref_paths, novel_paths)[1] for key, scorer in scorers.items()}
            
            # Populate the rows
            for i, (src_file, tgt_file, (obj_idx, src_view, tgt_view), features) in enumerate(zip(source_paths, novel_paths, object_num__source_view_index__target_view_index, source_novel_joint_features)):
                psnr = temp_scores['psnr'][i] if 'psnr' in temp_scores else None

                data_rows.append({
                    "source_file": src_file,
                    "target_file": tgt_file,
                    "object_index": obj_idx,
                    "object_name": object_index_to_object_name[obj_idx],
                    "source_view": src_view,
                    "target_view": tgt_view,
                    "feature": features,
                    "psnr": psnr,
                    "delta_azimuth": float(delta_azimuth_in_degrees.replace("delta",""))
                })
            

                
    if mast3r_flag:
        df = pd.DataFrame(data_rows)
        df.to_pickle(f"{exp_root_folder}/Mast3r_{mast3r_mode}.pkl")



if __name__ == "__main__":
    model_name = "zero123"
    exp_root_folder = os.path.join(home_directory,f"zero123/zero123/output_images/gso-30-360_{model_name}/")
    extract_features_from_exp(exp_root_folder, mast3r_args=[True,"pts3d_features2"])
