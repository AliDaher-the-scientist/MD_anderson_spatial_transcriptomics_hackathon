# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:56:07 2026

@author: ARDaher
"""

import os
cwd = os.getcwd()
# ✅ Set all environment variables BEFORE importing torch
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Optional: for better error traces

# ✅ Import torch early, then check for CUDA
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Now import the rest
import scanpy as sc
import cell2location
from cell2location.models import RegressionModel
import scvi

import numpy as np
from cell2location.models import Cell2location
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs
#from cell2location.utils import select_slide
from cell2location.plt import plot_spatial
import scipy
import scvi
#from cell2location.utils import select_slide
import seaborn as sns
from matplotlib.colors import to_hex


results_folder = os.path.join(cwd, 'results_5_20/analysis/')

# Reference paths remain the same
ref_run_name = f'{results_folder}/reference_signatures'
run_name = f'{results_folder}/cell2location_map'

# 1. Load Raw scRNA-seq Data
sc_adata = sc.read_h5ad('CRC_integrated_refined.h5ad')
sc_adata.layers["counts"] = sc_adata.layers["counts"].astype(int) #convert to integer
sc_adata.X = sc_adata.layers['counts'].copy()
# 2. Inspect the data
print("Checking sc_adata.X after loading:")
print(f"  Data type: {sc_adata.X.dtype}")
print(f"  Min value: {sc_adata.X.min()}")
print(f"  Max value: {sc_adata.X.max()}")

# 3. Add cell type labels (if needed) - Make sure 'lineage' is the right column name
print(sc_adata.obs.columns) # Check available columns
# If 'lineage' or cell type labels are missing, load them and add to sc_adata.obs

# 4. Perform Permissive Gene Filtering (scRNA-seq)
from cell2location.utils.filtering import filter_genes
selected = filter_genes(sc_adata, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
sc_adata = sc_adata[:, selected].copy()




# 6. Setup and Train the Reference Model (scRNA-seq)
# 1. Set environment variables for CPU optimization (BEFORE model init)
os.environ["OMP_NUM_THREADS"] = "4"          # For matrix operations
os.environ["MKL_NUM_THREADS"] = "4"          # For math kernel
os.environ["OPENBLAS_NUM_THREADS"] = "4"     # For BLAS
torch.utils.data.DataLoader.num_workers = 40  # Default worker count
print(f'Number of workers is {torch.utils.data.DataLoader.num_workers}')
scvi.settings.dl_num_workers = 40
print(f'Number of workers set to: {scvi.settings.dl_num_workers}')
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Training on GPU.")
# 2. Initialize and setup model (standard)

RegressionModel.setup_anndata(
    adata=sc_adata, 
    layer="counts", 
    labels_key="refined_cell_type"
)


# Set precision for matrix multiplications - important for Tensor Cores
torch.set_float32_matmul_precision('high')
# 3. Configure training with CPU-specific settings
sc_model = RegressionModel(sc_adata)
# Move the model to CUDA device if available
sc_model.train(
    max_epochs=500,
    train_size=1.0,   # Disables validation warnings
    accelerator='cpu',
    log_every_n_steps = 1,
    batch_size = 256, 
)




# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_ref = sc_model.export_posterior(
    sc_adata, sample_kwargs={'accelerator':"gpu"}
)

# Save model


# Save model
sc_model.save(f"{ref_run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref.write(adata_file)
#adata_file



#can be loaded later like this:
#adata_file = f"{ref_run_name}/sc.h5ad"
#adata_ref = sc.read_h5ad(adata_file)
#sc_model = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)

# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]




# 7. Load Raw Spatial Transcriptomics Data
st_adata_raw = sc.read_h5ad('GSM8594562_processed.h5ad')
st_adata_raw.var_names_make_unique()
# 8. Inspect data
print("Checking st_adata_raw.X after loading:")
print(f"  Data type: {st_adata_raw.X.dtype}")
print(f"  Min value: {st_adata_raw.X.min()}")
print(f"  Max value: {st_adata_raw.X.max()}")


# 9. Filter Spatial Data (Mitochondrial Genes)
st_adata_raw.var['MT_gene'] = [gene.startswith('MT-') for gene in st_adata_raw.var.index]
st_adata_raw.obsm['MT'] = st_adata_raw[:, st_adata_raw.var['MT_gene'].values].X.toarray()
st_adata_raw = st_adata_raw[:, ~st_adata_raw.var['MT_gene'].values]




# find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(st_adata_raw.var_names, inf_aver.index)
st_adata_raw = st_adata_raw[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=st_adata_raw)


# create and train the model
mod = cell2location.models.Cell2location(
    st_adata_raw, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=5,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection:
    detection_alpha=20
)
#mod.view_anndata_setup()
scvi.settings.dl_num_workers = 40
mod.train(max_epochs=30000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          accelerator="gpu",
          log_every_n_steps = 1
         )

# plot ELBO loss history during training, removing first 100 epochs from the plot
mod.plot_history(1000)
plt.legend(labels=['full data training']);

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
st_adata_raw = mod.export_posterior(
    st_adata_raw, sample_kwargs={'batch_size': mod.adata.n_obs,'accelerator':"gpu"}
)

# Save model
mod.save(f"{run_name}", overwrite=True)

# mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

# Save anndata object with results
adata_file = f"{run_name}/sp.h5ad"
st_adata_raw.write(adata_file)
adata_file


adata_vis = st_adata_raw.copy()
#can be uploaded later
#adata_file_spatial = f"{run_name}/sp.h5ad"
#adata_vis = sc.read_h5ad(adata_file_spatial)
#mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

mod.plot_QC() #should be roughly diagonalresults_folder = './results/keloid_analysis/'

# create paths and names to results folders for reference regression and cell2location models






# sample_id = 'Patient 2 CRC'  # Your sample name
# cell_types_of_interest = ['Myofibroblasts', 'CD19+CD20+ B', 'CD8+ T cells', 'IgG+ Plasma', 'NK cells', 'Regulatory T cells', 'CMS3', 'CMS4', 'Mast cells', 'cDC']
# clust_labels = ['Myofibroblasts', 'CD19+CD20+ B', 'CD8+ T cells', 'IgA+ Plasma', 'Regulatory T cells', 'CD3']  # For combined plot

# # ==============================================
# # 1. LOAD AND PREPARE SPATIAL DATA
# # ==============================================
# # Load scalefactors
# with open('GSM8594561_P2CRC_scalefactors_json.json') as f:
#     scalefactors = json.load(f)

# image = plt.imread("GSM8594561_P2CRC_image.tif")
# image_transposed = np.transpose(image, (1, 0, 2))  # Transpose H and W
# # Add image and metadata
# adata_vis.uns['spatial'] = {
#     sample_id: {
#         'images': {'hires': image_transposed},
#         'scalefactors': scalefactors
#     }
# }

# # Verify alignment parameters
# print(f"Spot diameter: {scalefactors['spot_diameter_fullres']} pixels")
# print(f"Tissue scale factor: {scalefactors['tissue_hires_scalef']}")




# """
# Plot q05
# """
# # add 5% quantile, representing confident cell abundance, 'at least this amount is present',
# # to adata.obs with nice names for plotting
# adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']


# # ==============================================
# # 2. PLOT CELL TYPES SEPARATELY (SCANPY)
# # ==============================================

     


# # select one slide
# #from cell2location.utils import select_slide
# # Select one slide for visualization
# slide = adata_vis

# # Plot each cell type of interest separately in spatial coordinates
# with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [6, 6]}):  # Adjust figure size
#     sc.pl.spatial(slide, cmap='RdYlBu_r',
#                   color=cell_types_of_interest,  # List of cell types to display
#                   ncols=3,  # Number of columns for the plot
#                   size=32 * scalefactors['tissue_hires_scalef'],  # Scale point size with tissue scale factor
#                   img_key='hires',  # Use 'hires' image key
#                   vmin=0, vmax='p99.2'  # Color scale set to the 99.2% quantile
#                  )


    

# # ==============================================
# # 3. PLOT CELL TYPES TOGETHER (CELL2LOCATION)
# # ==============================================
# clust_col = ['' + str(i) for i in clust_labels] # in case column names differ from labels
# palette = sns.color_palette("tab10", n_colors=len(clust_col))  # or use "Set3", "Paired", etc.
# color_dict = {k: to_hex(c) for k, c in zip(clust_col, palette)}

# # Assign colors to the `.uns` field for each column (cluster) in your data
# for col in clust_col:
#     slide.uns[col] = {'colors': color_dict}
    
# with mpl.rc_context({'figure.figsize': (15, 15)}):
#     fig = plot_spatial(
#         adata=slide,
#         # labels to show on a plot
#         color=clust_col, labels=clust_labels,
#         show_img=True,
#         # 'fast' (white background) or 'dark_background'
#         style='fast',
#         # limit color scale at 99.2% quantile of cell abundance
#         max_color_quantile=0.992,
#         # size of locations (adjust depending on figure size)
#         circle_diameter=7,
#         colorbar_position='right',
#     )


# """
# Plot mean
# """

# adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['means_cell_abundance_w_sf']


# # ==============================================
# # 2. PLOT CELL TYPES SEPARATELY (SCANPY)
# # ==============================================

     


# # select one slide
# #from cell2location.utils import select_slide
# # Select one slide for visualization
# slide = adata_vis

# # Plot each cell type of interest separately in spatial coordinates
# with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [6, 6]}):  # Adjust figure size
#     sc.pl.spatial(slide, cmap='RdYlBu_r',
#                   color=cell_types_of_interest,  # List of cell types to display
#                   ncols=3,  # Number of columns for the plot
#                   size=32 * scalefactors['tissue_hires_scalef'],  # Scale point size with tissue scale factor
#                   img_key='hires',  # Use 'hires' image key
#                   vmin=0, vmax='p99.2'  # Color scale set to the 99.2% quantile
#                  )


    

# # ==============================================
# # 3. PLOT CELL TYPES TOGETHER (CELL2LOCATION)
# # ==============================================
# clust_col = ['' + str(i) for i in clust_labels] # in case column names differ from labels
# palette = sns.color_palette("tab10", n_colors=len(clust_col))  # or use "Set3", "Paired", etc.
# color_dict = {k: to_hex(c) for k, c in zip(clust_col, palette)}

# # Assign colors to the `.uns` field for each column (cluster) in your data
# for col in clust_col:
#     slide.uns[col] = {'colors': color_dict}
    
# with mpl.rc_context({'figure.figsize': (15, 15)}):
#     fig = plot_spatial(
#         adata=slide,
#         # labels to show on a plot
#         color=clust_col, labels=clust_labels,
#         show_img=True,
#         # 'fast' (white background) or 'dark_background'
#         style='fast',
#         # limit color scale at 99.2% quantile of cell abundance
#         max_color_quantile=0.992,
#         # size of locations (adjust depending on figure size)
#         circle_diameter=7,
#         colorbar_position='right',
#     )


# """Save information in csv format"""
# adata_vis.obsm['means_cell_abundance_w_sf'].to_csv('mean_cell_abundances.csv')
# adata_vis.obsm['q05_cell_abundance_w_sf'].to_csv('q05_cell_abundances.csv')