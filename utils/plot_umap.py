import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from scipy import linalg
import torch
import sys
import pandas as pd
import pickle
import umap
from sklearn.preprocessing import StandardScaler
import umap.plot

np.random.seed(42)


import plotly.graph_objs as go

home_directory = os.path.expanduser('~')
# # Add the path to your local pytorch_fid directory to the beginning of sys.path
# sys.path.insert(0, os.path.join(home_directory,"pytorch-fid/src"))
# import pytorch_fid.fid_score

from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.plotting import output_file, save

index_to_object_name = {
    0: 'alarm',
    1: 'backacpk',
    2: 'bell',
    3: 'blocks',
    4: 'chicken',
    5: 'cream',
    6: 'elephant',
    7: 'grandfather',
    8: 'grandmother',
    9: 'hat',
    10: 'leather',
    11: 'lion',
    12: 'lunch_bag',
    13: 'mario',
    14: 'oil',
    15: 'school_bus1',
    16: 'school_bus2',
    17: 'shoe',
    18: 'shoe1',
    19: 'shoe2',
    20: 'shoe3',
    21: 'soap',
    22: 'sofa',
    23: 'sorter',
    24: 'sorting_board',
    25: 'stucking_cups',
    26: 'teapot',
    27: 'toaster',
    28: 'train',
    29: 'turtle'
}


###############################################################################################################

if __name__ == "__main__":
    # Step 0: Initializations

    file_folder = os.path.join(home_directory,"zero123/zero123/output_images/gso-30-360_zero123-xl")
    figures_path = "figures/gso-30-360_zero123-xl"

    # file_folder = "features"
    # figures_path = "figures"

    os.makedirs(figures_path, exist_ok=True)
    file_name = "Mast3r_pts3d_features2_whiteBG"
    file_path = os.path.join(file_folder, file_name + ".pkl")
    df = pd.read_pickle(file_path)

    # Step 1:  prepare data
    translate_indices_to_names = np.vectorize(index_to_object_name.get)
    df["object_name"] = translate_indices_to_names(df["object_index"])

    # # List of objects to filter out
    objects_to_filter = ['bell', 'cream', 'chicken', 'hat', 'soap', 'sorter', 'stucking_cups','toaster','sorting_board','lunch_bag']
    # objects_to_filter = []


    # Filter out the objects in the list
    df = df[~df["object_name"].isin(objects_to_filter)]
    df = df.reset_index(drop=True)


    hover_data = pd.DataFrame({
        'delta_azimuth': df["delta_azimuth"],
        'object_name': df["object_name"]
    })

    if 'psnr' in df.columns:
        hover_data['psnr'] = df['psnr']

    # Step 2: Apply UMAP to reduce to 2 dimensions
    scaled_features = StandardScaler().fit_transform(np.stack(df["feature"].values))
    n_components = 2
    n_neighbors = 60
    reducer = umap.UMAP(n_neighbors = n_neighbors, n_components = n_components)
    embedding = reducer.fit_transform(scaled_features)
    if n_components==2:
        output_file(os.path.join(figures_path, f"UMAP_{file_name}.html"), title=f'UMAP Dimension Reduction of {file_name}')
        p1 = umap.plot.interactive(reducer, labels=df["delta_azimuth"], hover_data=hover_data, point_size=2, theme="inferno")
        p1.legend.title = "Δ Azimuth (Degrees) Between Image Pairs"
        p2 = umap.plot.interactive(reducer, labels=df["object_name"], hover_data=hover_data, point_size=2, theme="inferno")
        p2.legend.title = "Object Name"
        plots = [p1, p2]
        save(row(*plots))
