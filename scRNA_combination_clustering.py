# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:49:55 2026

@author: ARDaher
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:08:42 2026

@author: raluca
"""
import gc
import glob
import matplotlib
import os
matplotlib.use('module://matplotlib_inline.backend_inline')
from scipy.io import mmread
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap
from scipy.sparse import save_npz
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from scipy.sparse import csr_matrix, issparse
import scanpy as sc
import anndata
import datetime
import scrublet as scr
#import diffxpy.api as de
#import singler
import matplotlib.pyplot as plt
import celltypist
import harmonypy as hm
from celltypist import models
from matplotlib import patheffects
import scanpy.external as sce

def create_annotated_adata_slice(adata, model):
    adata.var_names_make_unique()
    matrix = adata.X #express cells as rows, genes as columns
    genes = adata.var_names
    mito_genes = [gene for gene in range(len(genes)) if 'MT-' in genes[gene]]
    # mito_genes now holds the indices of genes that are mitochondrial


    """
    Apply first layer of filtering
    """
    # Assuming 'matrix' is the expression matrix (cells x genes)
    # Sum the expression of mitochondrial genes in each cell
    mito_genes_expression = np.sum(matrix[:,mito_genes], axis=1)
    # Now we can compute the percentage of mitochondrial gene expression per cell
    total_counts_expression_per_cell = np.sum(matrix, axis=1)  # Total counts per cell
    mito_percentage_per_cell = mito_genes_expression / total_counts_expression_per_cell  # Percentage of mitochondrial gene expression

    min_gene_number = 150  #this should be 200.
    max_gene_number = 8000
    min_expression_count=400
    umi_counts_per_cell = np.sum(matrix > 0, axis=1)  # Count of non-zero values (i.e. number genes) per cell
    total_counts_per_cell = np.sum(matrix, axis=1) #total count per cell
    filtered_cells = (umi_counts_per_cell > min_gene_number) & (umi_counts_per_cell < max_gene_number) &\
        (total_counts_per_cell > min_expression_count) & (mito_percentage_per_cell < 0.2)
    filtered_matrix = matrix[np.where(filtered_cells==True)[0],:]



    #Apply scrublet filtering to remove double cell detection
    scrub = scr.Scrublet(filtered_matrix)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()

    # Set the threshold for doublet detection (mean + 2 * absolute deviation)
    #threshold = np.mean(doublet_scores) + 2 * np.std(doublet_scores)
    #predicted_doublets = doublet_scores > threshold  # Cells above threshold are predicted doublets

    # Step 4: Exclude predicted doublets
    filtered_matrix_2 = filtered_matrix[np.where(predicted_doublets == 0)[0],:]  # Only include cells that are not doublets
    #filtered_matrix_2 = filtered_matrix_2.toarray()
    #Now final_filtered_matrix contains only the cells that passed the filtering criteria and doublet removal.


    """
    Optional: apply second layer of filtering
    """
    #filtered_matrix_3 = np.load("filtered_matrix_3.npy")

    # Check if it's already sparse
    if not issparse(filtered_matrix_2):
        filtered_matrix_2 = csr_matrix(filtered_matrix_2)  # Convert to CSR format



    umi_counts_per_cell = np.sum(filtered_matrix_2 > 0, axis=1)  # Count of non-zero values (i.e. number genes) per cell
    #umi_counts_per_cell = filtered_matrix_3.count_nonzero(axis=1)
    total_counts_per_cell = np.sum(filtered_matrix_2, axis=1) #total count per cell
    filtered_cells = (umi_counts_per_cell > min_gene_number) & \
        (umi_counts_per_cell < max_gene_number) &\
        (total_counts_per_cell > min_expression_count)
    filtered_matrix_3 = filtered_matrix_2[np.where(filtered_cells==True)[0],:]

    #Now filter out genes that are expressed in less than 10 cells
    cells_per_gene = np.array((filtered_matrix_3 > 0).sum(axis=0)).flatten() # Convert to 1D array
    # Apply filtering
    final_filtered_matrix = filtered_matrix_3[:, np.where(cells_per_gene > 10)[0]]
    # Step 1: Identify genes that survive the filter
    genes_kept = genes[np.where(cells_per_gene > 10)[0]]  # Indices of genes that survive first filtering

    adata = sc.AnnData(final_filtered_matrix)
    adata.var_names = genes_kept.astype(str).values  # Force string type
    adata.var_names_make_unique()  # Ensure unique names
    # To keep the original indices as well
    adata.layers["counts"] = adata.X.copy()
    #required for cell type annotation
    # Proceed with normalization and other preprocessing steps
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    
    sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor='seurat')




    if "highly_variable" not in adata.var.columns:
        raise ValueError("Highly variable genes were not computed correctly!")
        
        
    # 2. Identify HVGs

# 3. Calculate PCA and Neighbors ON ADATA 
# use_highly_variable=True makes this identical to your 'adata_hvg' math
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(
    adata, 
    resolution=10, 
    key_added='over_clustering'
)

# 4. Now run CellTypist - it will see the neighbors and WON'T crash
    pred = celltypist.annotate(
    adata,
    model,
    majority_voting=True,over_clustering='over_clustering')

    

# 5. Finish with Leiden/UMAP on the now-labeled adata
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8)
    adata = pred.to_adata()



    return adata


# 1. Find all GSM h5 files
file_list = sorted(glob.glob("GSM*.h5"))

preprocessed_adata_list = []
pool_lib = []
for file in file_list:
    print(f"Loading {file}...")
    temp_adata = sc.read_10x_h5(file)
    
    # 2. Extract 'P2L1' (the part between the 1st and 2nd underscore)
    # Filename: GSM8594537_P2L1_P1CRC_...
    # split('_') results in: ['GSM8594537', 'P2L1', 'P1CRC', ...]
    pool_lib_id = file.split('_')[1] 
    
    # Assign it to the metadata
    pool_lib.append( pool_lib_id)
    
    # Make gene names unique
    temp_adata.var_names_make_unique()
    
    preprocessed_adata_list.append(temp_adata)

print(f"Successfully loaded {len(preprocessed_adata_list)} datasets.")
post_processed_adata_list =[]
#pool_lib = ['P2L1', 'P2L2', 'P2L3']
CRC_model = 'Human_Colorectal_Cancer.pkl'
for i, local_adata in enumerate(preprocessed_adata_list):
    processed_adata = create_annotated_adata_slice(local_adata, CRC_model)
    #processed_adata_local.obs['pool_lib'] = pool_lib[i]
    post_processed_adata_list.append(processed_adata)


adata_all = sc.concat(post_processed_adata_list, label="pool_lib", keys=pool_lib, join="inner", index_unique = '_')            
            # Add metadata directly to the local object before appending


sc.pp.highly_variable_genes(adata_all, batch_key='pool_lib',n_top_genes=2000, flavor='seurat')


sc.pp.scale(adata_all, max_value=10)
sc.tl.pca(adata_all, use_highly_variable=True)
PCA_matrix = adata_all.obsm['X_pca']
pool_lib = adata_all.obs['pool_lib']
harmony_out = hm.run_harmony(PCA_matrix, adata_all.obs, "pool_lib",max_iter_harmony=20)
adata_all.obsm['X_pca_harmony'] = harmony_out.Z_corr
sc.pp.neighbors(adata_all,use_rep='X_pca_harmony')
sc.tl.umap(adata_all)


sc.pl.umap(adata_all, color='pool_lib', show = True)
plt.show()
sc.pl.umap(adata_all,color='majority_voting', legend_loc = 'on data',
           legend_fontsize=5, legend_fontoutline=2,frameon=False)

sc.tl.leiden(adata_all, resolution=1.0, key_added='leiden_clusters')
def refine_with_fallback(cluster_series):
    counts = cluster_series.value_counts()

    known_counts = counts.drop('Unknown', errors='ignore')
    
    # Only assign a name if known cells make up more than 5% of the cluster
    if not known_counts.empty and (known_counts.sum() / counts.sum() > 0.05):
        return known_counts.index[0]
    else:
        return 'Unknown'

# Apply this logic
cluster_to_label = adata_all.obs.groupby('leiden_clusters')['majority_voting'].agg(refine_with_fallback)
adata_all.obs['refined_cell_type'] = adata_all.obs['leiden_clusters'].map(cluster_to_label)

sc.pl.umap(adata_all,color='refined_cell_type', legend_loc = 'on data',
           legend_fontsize=5, legend_fontoutline=2,frameon=False)


# 1. Create the slim object
needed_obs = ['refined_cell_type', 'pool_lib']
adata_slim = adata_all[:, :].copy()

# 2. Keep only essential metadata
adata_slim.obs = adata_all.obs[needed_obs].copy()

# 3. FORCE counts and X to be sparse (This is the "magic" for size reduction)
if 'counts' in adata_slim.layers:
    adata_slim.layers['counts'] = csr_matrix(adata_slim.layers['counts'])
    
# Replace .X with the sparse counts to save even more space 
# (Since cell2location needs counts anyway)
adata_slim.X = csr_matrix(adata_slim.layers['counts'])

# 4. Wipe everything else
adata_slim.obsm = {}
adata_slim.obsp = {}
adata_slim.uns = {}
adata_slim.varm = {}

# 5. Save with compression (Essential for Jupyter uploads)
adata_slim.write("CRC_integrated_refined.h5ad", compression='gzip')

# Free up memory in your current session
gc.collect()

print("Saved slimmed adata with 'counts' layer and 'refined_cell_type'.")



#adata_all.write("CRC_integrated_refined.h5ad")
# Marker genes for your T-cell types

