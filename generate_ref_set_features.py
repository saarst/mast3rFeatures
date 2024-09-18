import csv
import os
import shutil
import sys
from mast3r_feature_extractor import MasterFeaturesExtractor
import pathlib
import numpy as np
import torch
import mast3r.utils.path_to_dust3r
from dust3r.utils.image import load_images

import pandas as pd
from utils.general import object_index_to_object_name, IMAGE_EXTENSIONS



home_directory = os.path.expanduser('~')


# Add the path to your local pytorch_fid directory to the beginning of sys.path
# sys.path.insert(0, os.path.join(home_directory,"pytorch-fid/src"))

# import pytorch_fid.fid_score
# from pytorch_fid.inception import InceptionV3
# Print the file path of the imported module to verify
# print(pytorch_fid.fid_score.__file__)

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    num_cpus = os.cpu_count()
num_workers = min(num_cpus, 8) if num_cpus is not None else 0


def get_image_files(data_path, num_views):
    """Retrieve and filter image files, excluding depth images."""
    return sorted(
        str(file)
        for ext in IMAGE_EXTENSIONS
        for file in pathlib.Path(data_path).rglob(f"*.{ext}")
        if "depth" not in str(file).lower() and any(f"{i:03d}" in str(file) for i in range(num_views))
    )


stage = 4


if stage == 1:                                  
    if data == "GSO-30-360":
        number_of_objects = 30
        num_views = 16
        batch_size = 60
        data_path = os.path.join(home_directory,"zero123/zero123/data/gso-30-360")

        if dust3r:
            output_file = os.path.join(data_path, f"Dust3r_single.npz")

            dust3r_extractor = Dust3rFeature(size=256, device=device, batch_size=batch_size, num_workers=num_workers)
            # Aggregate all the files
            files = sorted(
                [
                    str(file)
                    for ext in IMAGE_EXTENSIONS
                    for file in pathlib.Path(data_path).rglob("*.{}".format(ext))
                    if "depth" not in str(file).lower() and any(f"{i:03d}" in str(file) for i in range(num_views))
                ]
            )
            features = dust3r_extractor.extract_features(files)
            # features = dust3r_extractor.extract_single_images_features(files)
            arrays_dict = {str(i): features[i::num_views] for i in range(num_views)}
            # files_dict = {str(i): files[i::num_views] for i in range(num_views)}
            np.savez(output_file, **arrays_dict)
            # # validation = np.load(output_file)
            dust3r_extractor.to_cpu()
            del dust3r_extractor
        if inception:
            for dims in inception_dims:
                output_file = os.path.join(data_path, f"Inception_d{dims}.npz")
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                inception_model = InceptionV3([block_idx]).to(device)

                # Aggregate all the files
                files = sorted(
                    [
                        str(file)
                        for ext in IMAGE_EXTENSIONS
                        for file in pathlib.Path(data_path).rglob("*.{}".format(ext))
                        if "depth" not in str(file).lower() and any(f"{i:03d}" in str(file) for i in range(num_views))
                    ]
                )
                features = pytorch_fid.fid_score.get_activations(files, inception_model, batch_size=batch_size, dims=dims, device=device,
                            num_workers=num_workers)
                arrays_dict = {str(i): features[i::num_views] for i in range(num_views)}
                np.savez(output_file, **arrays_dict)
                # validation = np.load(output_file)
                inception_model.to('cpu')
                del inception_model
        if clip:
            batch_size = 32
            output_file = os.path.join(data_path, f"CLIP_single.npz")
            embedding_model = embedding.ClipEmbeddingModel()
            # Aggregate all the files
            files = sorted(
                [
                    str(file)
                    for ext in IMAGE_EXTENSIONS
                    for file in pathlib.Path(data_path).rglob("*.{}".format(ext))
                    if "depth" not in str(file).lower() and any(f"{i:03d}" in str(file) for i in range(num_views))
                ]
            )
            features = io_util.compute_embeddings_for_dir(files, embedding_model, batch_size, -1).astype("float32")
            # features = dust3r_extractor.extract_single_images_features(files)
            arrays_dict = {str(i): features[i::num_views] for i in range(num_views)}
            # files_dict = {str(i): files[i::num_views] for i in range(num_views)}
            np.savez(output_file, **arrays_dict)

if stage == 2: # concat per delta-azimuth
    data_path = os.path.join(home_directory,"zero123/zero123/data/gso-30-360")
    number_of_objects = 30
    num_views = 16
    data_files = [os.path.join(data_path, f"Inception_d{dim}.npz") for dim in inception_dims] + [os.path.join(data_path, f"Dust3r_single.npz")]
    output_files = [os.path.join(data_path, f"Inception_concat_d{dim}.npz") for dim in inception_dims] + [os.path.join(data_path, f"Dust3r_concat.npz")]
    if clip:
        data_files = [os.path.join(data_path, f"CLIP_single.npz")]
        output_files = [os.path.join(data_path, f"CLIP_concat.npz")]
    for data_file, output_file in zip(data_files, output_files):
        data = np.load(data_file)
        features_concat_dict = {str(d): np.vstack([np.hstack((data[str(i)], data[str((i + d) % num_views)])) 
                            for i in range(num_views)]) 
                            for d in range(16)}
        np.savez(output_file, **features_concat_dict)

if stage == 3: # joint features per delta-azimuth
        num_objects = 30
        num_views = 16
        batch_size = 60
        mode = "features2"
        data_path = os.path.join(home_directory,"zero123/zero123/data/gso-30-360")
        output_file = os.path.join(data_path, f"Dust3r_joint_{mode}.npz")
        dust3r_extractor = Dust3rFeature(size=256, device=device, batch_size=batch_size, num_workers=num_workers)
        # Aggregate all the files
        files = sorted(
            [
                str(file)
                for ext in IMAGE_EXTENSIONS
                for file in pathlib.Path(data_path).rglob("*.{}".format(ext))
                if "depth" not in str(file).lower() and any(f"{i:03d}" in str(file) for i in range(num_views))
            ]
        )
        images = load_images(files,size = 256)
        azimuth_dict = {str(i): images[i::num_views] for i in range(num_views)}
        delta_dict = {
                    str(j): [(azimuth_dict[str(i)][k], azimuth_dict[str((i + j) % num_views)][k]) for k in range(num_objects) for i in range(num_views)]
                        for j in range(num_views)    
                    }  
        pairs = [item for sublist in delta_dict.values() for item in sublist]
        features = dust3r_extractor.extract_features_from_pairs(pairs, mode=mode)
        # features = dust3r_extractor.extract_features_from_pairs_v2(pairs)
        split_indices = [len(delta_dict[key]) for key in delta_dict]
        split_arrays = np.split(features, np.cumsum(split_indices)[:-1])
        features_joint_dict = {key: split_arrays[i] for i, key in enumerate(delta_dict)}
        np.savez(output_file, **features_joint_dict)
        # # validation = np.load(output_file)
        dust3r_extractor.to_cpu()
        del dust3r_extractor

if stage == 4: # like stage 3 but in dataframe
    num_objects = 30
    num_views = 16
    batch_size = 60
    mode = "pts3d_features2"
    data_path = os.path.join(home_directory, "zero123/zero123/data/gso-30-360")
    output_file = os.path.join("features", f"Mast3r_{mode}.pkl")  # Save as pkl
    extractor = MasterFeaturesExtractor(size=256, device=device, batch_size=batch_size, num_workers=num_workers)
    
    # 1. Load images
    files = get_image_files(data_path, num_views)
    images = load_images(files, size=256)

    # 2. Create DataFrame
    data = []
    image_pairs = []
    for obj_index in range(num_objects):
        for source_view in range(num_views):
            for delta_azimuth in range(num_views):
                target_view = (source_view + delta_azimuth) % num_views
                data.append({
                    "source_file": files[obj_index * num_views + source_view],
                    "target_file": files[obj_index * num_views + target_view],
                    "object_index": obj_index,
                    "object_name": object_index_to_object_name[obj_index],
                    "delta_azimuth": delta_azimuth*(360/num_views),
                    "source_view": source_view,
                    "target_view": target_view,
                })
                source_image = images[obj_index * num_views + source_view]
                target_image = images[obj_index * num_views + target_view]
                image_pairs.append((source_image, target_image))

    df = pd.DataFrame(data)

    # 4. Feed the pairs to the feature extractor
    features = extractor.extract_features_from_pairs(image_pairs, mode=mode)

    # 5. Add features to the DataFrame
    df['feature'] = [np.array(f) for f in features]
    
    # Save DataFrame to a CSV file
    df.to_pickle(output_file)
    



