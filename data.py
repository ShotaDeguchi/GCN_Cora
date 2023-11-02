"""
prepare data
"""

import os

import numpy as np
import torch


def load_cora(path='./cora', device='cpu'):
    """
    The graph convolutional operation rquires normalize the adjacency matrix: D^(-1/2) * A * D^(-1/2). This step 
    scales the adjacency matrix such that the features of neighboring nodes are weighted appropriately during 
    aggregation. The steps involved in the renormalization trick are as follows:
        - Compute the degree matrix.
        - Compute the inverse square root of the degree matrix.
        - Multiply the inverse square root of the degree matrix with the adjacency matrix.
    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels) # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    E = edges.shape[0] # Number of edges
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix

    degree_mat = torch.sum(adj_mat, dim=1) # Compute the sum of each row in the adjacency matrix (degree matrix)
    degree_mat = torch.sqrt(1 / degree_mat) # Compute the reciprocal square root of the degrees
    degree_mat[degree_mat == float('inf')] = 0 # Handle division by zero cases
    degree_mat = torch.diag(degree_mat).to_sparse() # Convert the degree matrix to a sparse diagonal matrix

    adj_mat = degree_mat @ adj_mat @ degree_mat # Apply the renormalization trick

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)
