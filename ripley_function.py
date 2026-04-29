#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:06:42 2026

@author: raluca
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import json
from scipy.spatial import distance_matrix
import pickle as pkl
from collections import defaultdict




#To calculate the data spatial statistical measures


def calc_ripley(w_a, w_b, I_r, n_spot, A_spot):
    """
    w_a shouldnt include edge spots, and w_b should be all spots
    I_r would be size S_in, S_total 
    returns a single Ripley score
    """
    # if either cell type has zero total abundance, return 0
    if np.sum(w_a) == 0 or np.sum(w_b) == 0:
        return 0.0
    
    a = np.sum(np.outer(w_a, w_b) * I_r)
    b = n_spot * A_spot / (np.sum(w_a) * np.sum(w_b))
    return a * b
def calc_ripley_all_types(W,I_r, n_spot, A_spot, n_cell_types):
    ripley_cell_types = np.zeros((n_cell_types, n_cell_types))
    for i in range(n_cell_types):
        for j in range(n_cell_types):
            w_a = W[:,i]#
            w_b = W[:,j]
            ripley_cell_types[i,j] = calc_ripley(w_a, w_b, I_r, n_spot, A_spot)
    return ripley_cell_types



def extract_data(dataframe, st_adata, scalefactor,q05_numbers, sample, cell_types):
    with open(f'{dataframe}.pkl','rb') as f:
        neighbourhood_info = pkl.load(f)
    st_adata_raw = sc.read_h5ad(f'{st_adata}.h5ad')
    spatial_coordinates = st_adata_raw.obsm['spatial']
    # Load the JSON file
    with open(f'{scalefactor}_scalefactors_json.json', 'r') as f:
        json_data = json.load(f)
    spot_diameter_res = json_data['spot_diameter_fullres']
        
    q05_cell_numbers = pd.read_csv(f'{q05_numbers}.csv')
    q05_cell_numbers.columns.values[0] = 'Nucleotide_ID'
    are_identical = neighbourhood_info['ID'].equals(q05_cell_numbers['Nucleotide_ID'])
    print("Are the columns identical?", are_identical) 
    
    
    prefix_q05 = 'q05cell_abundance_w_sf_'


    # Initialize zero-filled DataFrames with same index as q05/q95
    cell_numbers_of_interest_q05 = pd.DataFrame(
        0.0, index=q05_cell_numbers.index, columns=cells_of_interest
    )


    # Fill in any cell types that actually exist in the CSVs
    for cell in cell_types:
        col_q05 = f"{prefix_q05}{cell}"
        if col_q05 in q05_cell_numbers.columns:
            cell_numbers_of_interest_q05[cell] = q05_cell_numbers[col_q05]


    # Convert to densities (cell count / spot area)
    #cell_densities_q05 = cell_numbers_of_interest_q05.to_numpy()/A_spot
    cell_numbers_q05 = cell_numbers_of_interest_q05.to_numpy()





    #n_cell_types = cell_numbers_q05.shape[1]


   # present_types = (np.sum(cell_numbers_q05,axis=0) > 0)

    # Mask where both types in the pair are present
    #pair_mask = present_types[:, None] & present_types[None, :]
    #pair_mask = pair_mask.astype(float)  # NumPy version of float
   # mask = pair_mask[:, :, np.newaxis]   # add a new axis for radius
    

    n_spot = len(neighbourhood_info)
    
    dist_matrix = distance_matrix(spatial_coordinates, spatial_coordinates)\
        *55/spot_diameter_res
        
    coords = spatial_coordinates * 55 / spot_diameter_res
    coords_round = np.round(coords, 6)

    coord_set = set(map(tuple, coords_round))
    n_spot = coords_round.shape[0]

    translation_weights = np.ones((n_spot, n_spot), dtype=float)
    
    
    displacement_weights = {}
    
    for i in range(n_spot):
        for j in range(n_spot):
            dx, dy = coords_round[j] - coords_round[i]
            key = (dx, dy)
    
            if key not in displacement_weights:
                shifted_coords = np.round(coords_round + np.array([dx, dy]), 6)
                shifted_keys = map(tuple, shifted_coords)
    
                overlap_count = sum(k in coord_set for k in shifted_keys)
    
                if overlap_count > 0:
                    displacement_weights[key] = n_spot / overlap_count
                else:
                    displacement_weights[key] = 0.0
    
            translation_weights[i, j] = displacement_weights[key]
                
    L_c = np.min(dist_matrix[dist_matrix>0]) #m 
    A_spot = np.sqrt(3)/2*L_c**2 #m^2 
    print(f"L_c is {L_c} m")
    delta = 4
    ripley_radii_array = L_c * np.array([0,1,np.sqrt(3),2,np.sqrt(7),np.sqrt(8),3, 2*np.sqrt(3), np.sqrt(13), 4]) 
    ripley_radii = list(ripley_radii_array+delta)
                
    #neighbours = neighbourhood_info['Arrays'].tolist()
    #edge_spots = [i for i, arr in enumerate(neighbours) if len(arr) < 6]
    #all_spots = np.arange(len(neighbours) ).astype(int)
    #internal_spots_list = [list(all_spots)]
    I_r_r = [np.eye(n_spot)]
   # relevant_distances = dist_matrix[edge_spots,:]
    for i in range(len(ripley_radii)-1):
       # boolean_matrix = relevant_distances<ripley_radii[i]
        #exclude_spots = np.where(np.any(boolean_matrix, axis=0))[0]
        #internal_spots_list.append(list(np.setdiff1d(all_spots,exclude_spots).astype(int)))
        I_r = dist_matrix < ripley_radii[i+1]
        I_r_corrected = I_r.astype(float) * translation_weights
        I_r_r.append(I_r_corrected)
        
        
    return cell_numbers_q05, I_r_r, n_spot, A_spot, cell_types
    
    

    # Access the 'resolutions' key (if it exists)


def calc_ripley_all_r(W, I_r_r, n_spot, A_spot , n_cell_types):
    r_num = len(I_r_r)
    ripley_array = np.zeros((n_cell_types, n_cell_types, r_num))
    
    for i in range(r_num):
        I_r = I_r_r[i]
       # internal_spots = internal_spots_list[i]
        #I_r = I_r[internal_spots,:] #so that I_r list only depend on distance matrix and r applied to each element of the list
        ripley_matrix = calc_ripley_all_types(W, I_r, n_spot, A_spot, n_cell_types)
        ripley_array[:,:,i] = ripley_matrix

    return ripley_array



    
q05_cell_numbers, I_r_r, n_spot, A_spot, cell_types = extract_data()
n_cell_types = len(cell_types)
ripley_array = calc_ripley_all_r(q05_cell_numbers, I_r_r, n_spot, A_spot, n_cell_types)








    
    
    






    
    
    



cells_of_interest = ('FB-I', 'FB-II', 'FB-III', 'Mono-Mac')

# Prefixes







q05_ripley_array = calc_ripley_all_r(cell_numbers_q05, I_r_r, internal_spots_list)






cell_list = cells_of_interest
n_cell_types = len(cell_list)
r_um = np.array(ripley_radii)  *1e6  # x-axis values

for i in range(n_cell_types):
    for j in range(n_cell_types):
        q05_line = q05_ripley_array[i, j, :]
        q95_line = q95_ripley_array[i, j, :]

        cell_type_A = cell_list[i]  # "center" spots
        cell_type_B = cell_list[j]  # "surrounding" spots

        plt.figure(figsize=(5,4))
        plt.plot(r_um, q05_line*1e6, 'o-', label='q05')
        plt.plot(r_um, q95_line*1e6, 'o-', label='q95')
        plt.xlabel('Radius ($\mu$m)')
        plt.ylabel('Ripley Score ($mm^2$)')
        plt.title(f"Ripley score for {cell_type_B} surrounding {cell_type_A} at day {day}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        


max_q05 = np.max(cell_densities_q05, axis=0)
min_q05 = np.min(cell_densities_q05, axis=0)
max_q95 = np.max(cell_densities_q95, axis=0)
min_q95 = np.min(cell_densities_q95, axis=0)

mean_densities = cell_densities_mean
mean_neighbours_densities = np.zeros_like(mean_densities)
var_densities = cell_densities_stds**2
var_neighbours_densities = np.zeros_like(mean_densities)

for i in range(len(neighbours)):
    N = 1 + len(neighbours[i])
    mean_neighbour_density = 1/N* ( mean_densities[i] + np.sum(mean_densities[neighbours[i]]))
    var =  1/N**2 * (var_densities[i] + np.sum(var_densities[neighbours[i]]))
    mean_neighbours_densities[i] = mean_neighbour_density
    var_neighbours_densities[i] = var 

np.savez(
    f"spatial_data_day_{day}.npz",
    q05_ripley_array=q05_ripley_array,
    q95_ripley_array=q95_ripley_array,
    mask=mask,
    max_q05=max_q05,
    min_q05=min_q05,
    max_q95=max_q95,
    min_q95=min_q95,
    mean_densities = mean_densities,
    mean_neighbours_densities = mean_neighbours_densities,
    var_densities = var_densities,
    var_neighbours_densities = var_neighbours_densities
)