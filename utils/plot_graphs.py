from calc_distance import distances
# from general import object_index_to_object_name, object_name_to_object_index
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.colors as pc
home_directory = os.path.expanduser('~')
from scipy.spatial.distance import pdist, cdist
# from sklearn.preprocessing import StandardScaler


def save_results_to_csv(delta_azimuth_vec, frechet_dist_vec, mmd_vec, output_file):
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Theta': delta_azimuth_vec,
        'MMD': mmd_vec
    }).set_index('Theta').T  # Transpose to have 'Theta' as columns and distances as rows
    
    # Save to CSV
    results_df.to_csv(output_file)
    print(f"Results saved to {output_file}")

###############################################################################################################

if __name__ == "__main__":
    metric = "Mast3r_pts3d_features2_whiteBG"
    ref_features_file = f"features/{metric}.pkl"
    
    ref_df = pd.read_pickle(ref_features_file)

    # filter objects
    # objects_to_filter = ['bell', 'chicken', 'stucking_cups', 'cream', 'sorter']
    # objects_to_filter = ['bell', 'chicken', 'stucking_cups', 'cream', 'sorting_board', 'sorter', 'toaster', 'lunch_bag', 'teapot']
    objects_to_filter = ['bell', 'cream', 'chicken', 'hat', 'soap', 'sorter', 'stucking_cups','toaster','sorting_board','lunch_bag', 'teapot']

    # objects_to_filter = []


    # Filter out the objects in the list
    ref_df = ref_df[~ref_df["object_name"].isin(objects_to_filter)]
    ref_df = ref_df.reset_index(drop=True)


    # scaler:
    # Extract features into a (N, d) matrix
    # features_matrix = np.vstack(ref_df["feature"].to_numpy())

    # scaled_features_matrix = StandardScaler().fit_transform(features_matrix)

    # # Assign the scaled features back to the DataFrame
    # ref_df["feature"] = list(scaled_features_matrix)


    # # sigma rule of thumb
    # distances_within_X = pdist(np.vstack(ref_df["feature"].to_numpy()), 'euclidean')
    # sigma = np.median(distances_within_X)
    # print(sigma)

    # graph_mode = "single_set_all_objects"
    # graph_mode = "two_sets"
    graph_mode = "single_set_per_object"

    if graph_mode == "two_sets":
        model_names = ["zero123", "zero123-xl"]  # List of model names
        novel_features_base_path = os.path.join(home_directory, "zero123/zero123/output_images")
        figures_path = "figures/models_comparison_filtered"
        os.makedirs(figures_path, exist_ok=True)

        # Initialize containers for results
        mmd_rbf_results = {}
        mmd_poly_results = {}
        frechet_results = {}
        delta_azimuths_per_model = {}   
        for model_name in model_names:

            novel_features_file = os.path.join(novel_features_base_path, f"gso-30-360_{model_name}/{metric}.pkl")
            novel_df = pd.read_pickle(novel_features_file)
        
            # Filter out the objects in the list
            novel_df = novel_df[~novel_df["object_name"].isin(objects_to_filter)]
            novel_df = novel_df.reset_index(drop=True)

            frechet_dist_vec = []
            mmd_rbf_vec = []
            mmd_poly_vec = []
            model_delta_azimuths = []
            for delta_azimuth in novel_df["delta_azimuth"].unique():
                model_delta_azimuths.append(delta_azimuth)

                ref_df_delta = ref_df[ref_df["delta_azimuth"] == delta_azimuth]
                novel_df_delta = novel_df[novel_df["delta_azimuth"] == delta_azimuth]

                # Convert to (N,d) ndarray
                ref_features = np.vstack(ref_df_delta["feature"].to_numpy())
                novel_features = np.vstack(novel_df_delta["feature"].to_numpy())

                # # sigma rule of thumb
                # combined_features = np.vstack((ref_features, novel_features))
                # distances_within = pdist(combined_features, metric='euclidean')
                # sigma = np.median(distances_within)
                # print(sigma)

                # Calculate distances
                frechet_dist, mmd_rbf, mmd_poly = distances(ref_features, novel_features)
                frechet_dist_vec.append(frechet_dist)
                mmd_rbf_vec.append(mmd_rbf)
                mmd_poly_vec.append(mmd_poly)
            
            # Store results for the current model
            mmd_rbf_results[model_name] = mmd_rbf_vec
            mmd_poly_results[model_name] = mmd_poly_vec
            frechet_results[model_name] = frechet_dist_vec
            delta_azimuths_per_model[model_name] = model_delta_azimuths

        # Plot MMD RBF results for all models, using only available delta_azimuth values for each model
        plt.figure(figsize=(10, 6))
        for model_name, mmd_vec in mmd_rbf_results.items():
            plt.plot(delta_azimuths_per_model[model_name], mmd_vec, label=f"MMD - {model_name}", marker='o')
        plt.xlabel("Δ Azimuth (°) Between Image Pairs")
        plt.ylabel("MMD")
        plt.title(f"MMD-RBF ({metric}) Between Distributions as a Function of Δ Azimuth")
        plt.legend()
        plt.grid(True)
        mmd_plot_filename = os.path.join(figures_path, f"mmd_RBF_comparison_{metric}.png")
        plt.savefig(mmd_plot_filename)
        plt.close()
        print(f"MMD RBF plot saved to {mmd_plot_filename}")

        # Plot MMD RBF results for all models, using only available delta_azimuth values for each model
        plt.figure(figsize=(10, 6))
        for model_name, mmd_vec in mmd_poly_results.items():
            plt.plot(delta_azimuths_per_model[model_name], mmd_vec, label=f"MMD - {model_name}", marker='o')
        plt.xlabel("Δ Azimuth (°) Between Image Pairs")
        plt.ylabel("MMD")
        plt.title(f"MMD-poly ({metric}) Between Distributions as a Function of Δ Azimuth")
        plt.legend()
        plt.grid(True)
        mmd_plot_filename = os.path.join(figures_path, f"mmd_poly_comparison_{metric}.png")
        plt.savefig(mmd_plot_filename)
        plt.close()
        print(f"MMD poly plot saved to {mmd_plot_filename}")

        
        # Plot Frechet results for all models, using only available delta_azimuth values for each model
        plt.figure(figsize=(10, 6))
        for model_name, frechet_dist_vec in frechet_results.items():
            plt.plot(delta_azimuths_per_model[model_name], frechet_dist_vec, label=f"Frechet - {model_name}", marker='s')
        plt.xlabel("Δ Azimuth (°) Between Image Pairs")
        plt.ylabel("Frechet Distance")
        plt.title(f"Frechet Distance ({metric}) Between Distributions as a Function of Δ Azimuth")
        plt.legend()
        plt.grid(True)
        frechet_plot_filename = os.path.join(figures_path, f"frechet_comparison_{metric}.png")
        plt.savefig(frechet_plot_filename)
        plt.close()
        print(f"Frechet plot saved to {frechet_plot_filename}")

        # # Optionally, save the results to CSV files if needed
        # for model_name in model_names:
        #     csv_filename = os.path.join(figures_path, f"distance_{model_name}_{metric}.csv")
        #     save_results_to_csv(delta_azimuths_per_model[model_name], frechet_results[model_name], mmd_results[model_name], csv_filename)
        #     print(f"Results for {model_name} saved to {csv_filename}")




    if graph_mode == "single_set_per_object":
        figures_path = "figures/delta_figures_per_object_filtered"
        os.makedirs(figures_path, exist_ok=True)
        for delta_azimuth_ref in ref_df["delta_azimuth"].unique():
            ref_df_delta_ref = ref_df[ref_df["delta_azimuth"] == delta_azimuth_ref]
            mmd_rbf_dict = {}
            for object_name in ref_df["object_name"].unique():
                mmd_rbf_dict[object_name] = []
            delta_azimuth_vec = []
            for delta_azimuth_target in ref_df["delta_azimuth"].unique():
                ref_df_delta_target = ref_df[ref_df["delta_azimuth"] == delta_azimuth_target]
                delta_azimuth_vec.append(delta_azimuth_target)

                for object_name in ref_df["object_name"].unique():
                    ref_features = np.vstack(ref_df_delta_ref[ref_df_delta_ref["object_name"] == object_name]["feature"].to_numpy())
                    target_features = np.vstack(ref_df_delta_target[ref_df_delta_target["object_name"] == object_name]["feature"].to_numpy())

                    # Calculate MMD (only MMD, no Frechet)
                    _, mmd_rbf, _= distances(ref_features, target_features)
                    mmd_rbf_dict[object_name].append(mmd_rbf)
                
            unique_colors = pc.sample_colorscale('Viridis', [i / len(mmd_rbf_dict) for i in range(len(mmd_rbf_dict))])

            # Create an empty figure
            fig = go.Figure()

            # Loop through each object and add a trace (line) to the plot, assigning a unique color
            for idx, (object_name, mmd_vec) in enumerate(mmd_rbf_dict.items()):
                fig.add_trace(go.Scatter(
                    x=delta_azimuth_vec,
                    y=mmd_vec,
                    mode='lines+markers',
                    name=f'MMD for {object_name}',  # This will appear on hover
                    hoverinfo='name+x+y',  # Shows object name, x, and y on hover
                    line=dict(color=unique_colors[idx]),  # Assign unique color
                ))

            # Update layout to add titles, axes labels, etc.
            fig.update_layout(
                title=f"Feature Distance ({metric}): Reference Δ Azimuth ({delta_azimuth_ref}°) vs. Varying Δ Azimuth",
                xaxis_title='Varying Δ Azimuth (°) Between Image Pairs',
                yaxis_title='MMD',
                legend_title='Objects',
                hovermode='x',  # Allows hovering over x-axis values
                width=1000,  # Adjust the plot width
                height=600   # Adjust the plot height
            )


            # Optionally, save the plot as an interactive HTML file
            fig.write_html(os.path.join(figures_path, f"MMD_128_ref{delta_azimuth_ref}.html"))

    if graph_mode == "single_set_all_objects":
        figures_path = f"figures/delta_figures_{metric}_filtered"
        os.makedirs(figures_path, exist_ok=True)
        for delta_azimuth_ref in ref_df["delta_azimuth"].unique():
            ref_df_delta_ref = ref_df[ref_df["delta_azimuth"] == delta_azimuth_ref]
            frechet_dist_vec = []
            mmd_rbf_vec = []
            mmd_poly_vec = []
            delta_azimuth_vec = []
            for delta_azimuth_target in ref_df["delta_azimuth"].unique():
                ref_df_delta_target = ref_df[ref_df["delta_azimuth"] == delta_azimuth_target]

                # Convert to (N,d) ndarray
                ref_features = np.vstack(ref_df_delta_ref["feature"].to_numpy())
                target_features = np.vstack(ref_df_delta_target["feature"].to_numpy())

                # Calculate distances
                frechet_dist, mmd_rbf, mmd_poly = distances(ref_features, target_features)
                frechet_dist_vec.append(frechet_dist)
                mmd_rbf_vec.append(mmd_rbf)
                mmd_poly_vec.append(mmd_poly)
                delta_azimuth_vec.append(delta_azimuth_target)

                
                # Plotting both Frechet Distance and MMD on the same graph
                fig, ax1 = plt.subplots(figsize=(10, 6))  # Increase figure size

                # Plot Frechet Distance
                ax1.plot(delta_azimuth_vec, frechet_dist_vec, 'b-o', label='Frechet Distance')
                ax1.set_xlabel('Varying Δ Azimuth (°) Between Image Pairs')
                ax1.set_ylabel('Frechet Distance', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True)

                # Create a second y-axis for MMD
                ax2 = ax1.twinx()
                ax2.plot(delta_azimuth_vec, mmd_rbf_vec, 'r-o', label='MMD')
                ax2.set_ylabel('MMD-RBF', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # Title and saving the figure
                plt.title(f"Feature Distance ({metric}): Reference Δ Azimuth ({delta_azimuth_ref}°) vs. Varying Δ Azimuth")
                fig.tight_layout()  # Adjust layout to prevent overlap
                plt.savefig(os.path.join(figures_path, f"distance_{metric}_ref{delta_azimuth_ref}.png"))
                plt.close()