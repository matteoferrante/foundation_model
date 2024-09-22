import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import tqdm

from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def evaluate_reconstructions(ground_truths,reconstructions):
    # Initialize accumulators for the metrics
    r2_scores = []
    pearson_correlations = []
    mse_scores = []

    # Loop over each sample in the batch
    for i in range(ground_truths.shape[0]):
        # Flatten the ground truth and reconstructed series for the current sample
        ground_truth_flat = ground_truths[i].view(-1).cpu().numpy()
        reconstruction_flat = reconstructions[i].view(-1).cpu().numpy()

        # Compute R² for the current sample
        r2 = r2_score(ground_truth_flat, reconstruction_flat)
        r2_scores.append(r2)

        # Compute Pearson correlation for the current sample
        pearson_corr, _ = pearsonr(ground_truth_flat, reconstruction_flat)
        pearson_correlations.append(pearson_corr)

        # Compute MSE for the current sample
        mse = F.mse_loss(reconstructions[i], ground_truths[i]).item()
        mse_scores.append(mse)

    # Compute the average of each metric
    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_pearson = sum(pearson_correlations) / len(pearson_correlations)
    avg_mse = sum(mse_scores) / len(mse_scores)

    # Print the average results
    print(f'Average R² Score: {avg_r2:.4f}')
    print(f'Average Pearson Correlation: {avg_pearson:.4f}')
    print(f'Average Mean Squared Error (MSE): {avg_mse:.4f}')
    
    return avg_r2, avg_pearson, avg_mse


def get_model_output(model,loader, mask_ratio=None):
    # Lists to store the outputs, ground truth, and masks
    reconstructions = []
    ground_truths = []
    masks = []
    latents = []
    labels = []

    # Set the model to evaluation mode
    model.eval()

    # Test loop to extract ground truth, reconstructions, and masks
    with torch.no_grad():  # Disable gradient calculations for inference
        pbar = tqdm.tqdm(loader)

        for x, y in pbar:
            x = x

            # Forward pass to get the reconstruction and mask
            output, z, mask = model(x)

            # Trim output to match ground truth sequence length
            effective_seq_len = output.shape[1]
            data_trimmed = x[:, :effective_seq_len, :]

            # Append the results to the lists (detach from GPU and convert to CPU tensors)
            reconstructions.append(output.cpu())
            ground_truths.append(data_trimmed.cpu())
            latents.append(z.cpu())
            masks.append(mask.cpu())
            labels.append(y)
            
    reconstructions = torch.cat(reconstructions,0)
    ground_truths = torch.cat(ground_truths,0)
    masks = torch.cat(masks,0)
    latents = torch.cat(latents,0)
    labels = torch.cat(labels,0)
    
    
    
    return reconstructions, ground_truths, masks, latents, labels


