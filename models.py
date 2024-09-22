import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup  # You need to install the transformers library
from torch.optim.lr_scheduler import LambdaLR

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: Tensor of shape (sequence_length, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MaskedAutoencoder(nn.Module):


        """
    A Masked Autoencoder model designed for time series data, based on a Transformer encoder-decoder architecture.
    This model uses Conv1d for patch projection and ConvTranspose1d for patch reconstruction. It introduces masking
    during training to encourage the model to reconstruct missing parts of the input sequence.

    Attributes:
        patch_size (int): The size of each patch of the input time series.
        d_in (int): The dimensionality of the input time series.
        d_model (int): The dimensionality of the embedding space (used for Transformer layers).
        encoder_projection (nn.Conv1d): Conv1d layer that projects input patches into embeddings.
        positional_encoding (PositionalEncoding): Positional encoding to add information about the position of patches.
        transformer_encoder (nn.TransformerEncoder): A stack of Transformer encoder layers.
        transformer_decoder (nn.TransformerDecoder): A stack of Transformer decoder layers.
        decoder_projection (nn.ConvTranspose1d): ConvTranspose1d layer that projects embeddings back to the original time series space.

    Methods:
        patchify(x):
            Splits the input sequence into patches.
        
        depatchify(x_patches):
            Reconstructs the original sequence from patches.
        
        mask_patches(x_patches, mask_ratio=0.5):
            Randomly masks a proportion of the patches in the input sequence.
        
        forward(x, mask_ratio=0.5):
            Passes the input through the masked autoencoder and returns the reconstructed sequence, the latent representation, and the mask used.

    """
    def __init__(self, patch_size, d_in, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        
        """
        Initializes the MaskedAutoencoder.

        Args:
            patch_size (int): The size of patches into which the input sequence is split.
            d_in (int): Dimensionality of the input time series.
            d_model (int): Dimensionality of the embeddings used in the Transformer.
            nhead (int): Number of heads in the multi-head attention mechanism.
            num_encoder_layers (int): Number of layers in the Transformer encoder.
            num_decoder_layers (int): Number of layers in the Transformer decoder.
            dim_feedforward (int): Dimensionality of the feedforward layers in the Transformer.
            dropout (float): Dropout rate applied to the Transformer layers and positional encoding.
        """
        
        super(MaskedAutoencoder, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model  # Embedding dimension
        self.d_in = d_in
        
        # Shared Conv1d layer to project patches into embeddings
        self.encoder_projection = nn.Conv1d(in_channels=d_in, out_channels=d_model, kernel_size=patch_size)
        
        # Positional Encoding for the transformer
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Projection back to time series using ConvTranspose1d
        self.decoder_projection = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_in,
            kernel_size=patch_size
        )
        
    def patchify(self, x):

        """
        Splits the input sequence into patches.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Patches of shape (batch_size, n_patches, patch_size, d_in).
        """
        batch_size, seq_len, d_in = x.shape
        ps = self.patch_size
        n_patches = seq_len // ps  # Number of full patches
        x = x[:, :n_patches * ps, :]  # Trim to fit full patches
        x_patches = x.view(batch_size, n_patches, ps, d_in)
        return x_patches
    
    def depatchify(self, x_patches):

        """
        Reconstructs the original sequence from patches.

        Args:
            x_patches (torch.Tensor): Patches of shape (batch_size, n_patches, patch_size, d_in).

        Returns:
            torch.Tensor: Reconstructed sequence of shape (batch_size, seq_len, d_in).
        """


        batch_size, n_patches, ps, d_in = x_patches.shape
        x_reconstructed = x_patches.view(batch_size, n_patches * ps, d_in)
        return x_reconstructed
    
    def mask_patches(self, x_patches, mask_ratio=0.5):

        """
        Randomly masks a proportion of the patches in the input sequence.

        Args:
            x_patches (torch.Tensor): Input patches of shape (batch_size, n_patches, patch_size, d_in).
            mask_ratio (float): The proportion of patches to be masked.

        Returns:
            torch.Tensor: Masked patches.
            torch.Tensor: Mask used to mask the patches.
        """
        batch_size, n_patches, ps, d_in = x_patches.shape
        num_masked_patches = int(mask_ratio * n_patches)
        
        # Generate random masks
        random_scores = torch.rand(batch_size, n_patches, d_in, device=x_patches.device)
        # Determine threshold for masking
        threshold = torch.kthvalue(random_scores, num_masked_patches, dim=1).values.unsqueeze(1)
        mask = (random_scores < threshold).float()
        
        # Apply mask
        mask_expanded = mask.unsqueeze(2)  # Shape: (batch_size, n_patches, 1, d_in)
        x_masked = x_patches.clone()
        x_masked = x_masked * (1 - mask_expanded)
        
        return x_masked, mask
    
    def forward(self, x, mask_ratio=0.5):

        """
        Forward pass through the MaskedAutoencoder.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, d_in).
            mask_ratio (float): The proportion of patches to be masked.

        Returns:
            torch.Tensor: Reconstructed sequence.
            torch.Tensor: Latent representation of the sequence.
            torch.Tensor: Mask used during training.
        """

        # Patchify
        x_patches = self.patchify(x)  # Shape: (batch_size, n_patches, patch_size, d_in)
        
        # Mask patches
        x_masked_patches, mask = self.mask_patches(x_patches, mask_ratio=mask_ratio)
        
        batch_size, n_patches, ps, d_in = x_masked_patches.shape
        
        # Reshape for Conv1d (this is a trick to apply the same Conv1d to all patches)
        x_patches_reshaped = x_masked_patches.view(batch_size * n_patches, ps, d_in)  # Shape: (batch_size * n_patches, patch_size, d_in)
        x_patches_reshaped = x_patches_reshaped.permute(0, 2, 1)  # Shape: (batch_size * n_patches, d_in, patch_size)
        
        # Apply Conv1d to project patches into embeddings
        embeddings = self.encoder_projection(x_patches_reshaped)  # Shape: (batch_size * n_patches, d_model, output_length)
        embeddings = embeddings.squeeze(-1)  # Remove the last dimension if kernel_size == patch_size, output_length == 1
        
        # Reshape embeddings for the transformer
        embeddings = embeddings.view(batch_size, n_patches, self.d_model)  # Shape: (batch_size, n_patches, d_model)
        embeddings = embeddings.permute(1, 0, 2)  # Shape: (n_patches, batch_size, d_model)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Transformer Encoder
        z = self.transformer_encoder(embeddings)  # Shape: (n_patches, batch_size, d_model)
        
        # Transformer Decoder
        decoded = self.transformer_decoder(z, z)  # Shape: (n_patches, batch_size, d_model)
        
        # Prepare for ConvTranspose1d (same trick as before)
        decoded = decoded.permute(1, 0, 2).contiguous()  # Shape: (batch_size, n_patches, d_model)
        decoded = decoded.view(batch_size * n_patches, self.d_model).unsqueeze(-1)  # Shape: (batch_size * n_patches, d_model, 1)
        
        # Apply ConvTranspose1d to reconstruct patches
        reconstructed_patches = self.decoder_projection(decoded)  # Shape: (batch_size * n_patches, d_in, patch_size)
        
        reconstructed_patches = reconstructed_patches.contiguous()
        
        # Reshape to original format
        reconstructed_patches = reconstructed_patches.view(batch_size, n_patches, d_in, self.patch_size)
        reconstructed_patches = reconstructed_patches.permute(0, 1, 3, 2).contiguous()  # Shape: (batch_size, n_patches, patch_size, d_in)
        
        # Depatchify
        x_reconstructed = self.depatchify(reconstructed_patches)
        
        return x_reconstructed, z.permute(1,0,2), mask 



class MaskedAutoencoderLightning(pl.LightningModule):
    def __init__(self, 
                 patch_size=20, 
                 d_in=25, 
                 d_model=128, 
                 nhead=4, 
                 num_encoder_layers=4, 
                 num_decoder_layers=2, 
                 dim_feedforward=256, 
                 dropout=0.1,
                 learning_rate=1e-4,
                 weight_decay=0,
                 masked_loss=False,
                 mask_ratio=0.5,
                 warmup_steps=1000,
                 max_steps=10000,
                use_scheduler = False):
        super().__init__()
        self.save_hyperparameters()

        self.model = MaskedAutoencoder(
            patch_size=patch_size,
            d_in=d_in,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.criterion = nn.MSELoss()
        self.masked_loss = masked_loss
        self.mask_ratio = mask_ratio

    def forward(self, x, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        return self.model(x, mask_ratio=mask_ratio)

    def patchify(self, x):
        return self.model.patchify(x)

    def depatchify(self, x_patches):
        return self.model.depatchify(x_patches)
    
    def mask_patches(self, x_patches, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        return self.model.mask_patches(x_patches, mask_ratio=mask_ratio)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Assuming the batch is a tuple of (input, target), target is unused here
        batch_size = x.size(0)

        # Forward pass
        output, z, mask = self(x)

        # Compute loss
        effective_seq_len = output.shape[1]
        data_trimmed = x[:, :effective_seq_len, :]

        if self.masked_loss:
            # Compute loss only over masked patches
            mask_expanded = mask.unsqueeze(2).repeat(1, 1, self.hparams.patch_size, 1)  # Shape: (batch_size, n_patches, patch_size, d_in)
            mask_flat = mask_expanded.view(batch_size, -1, self.hparams.d_in)

            # Trim data and output
            data_masked = data_trimmed * mask_flat
            output_masked = output * mask_flat

            loss = self.criterion(output_masked, data_masked)
        else:
            loss = self.criterion(output, data_trimmed)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, prog_bar=True, logger=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)

        # Forward pass
        output, z, mask = self(x)

        # Compute loss
        effective_seq_len = output.shape[1]
        data_trimmed = x[:, :effective_seq_len, :]

        if self.masked_loss:
            # Compute loss only over masked patches
            mask_expanded = mask.unsqueeze(2).repeat(1, 1, self.hparams.patch_size, 1)
            mask_flat = mask_expanded.view(batch_size, -1, self.hparams.d_in)

            # Trim data and output
            data_masked = data_trimmed * mask_flat
            output_masked = output * mask_flat

            loss = self.criterion(output_masked, data_masked)
        else:
            loss = self.criterion(output, data_trimmed)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.use_scheduler:
            # Scheduler with warm-up and cosine annealing
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_steps - self.hparams.warmup_steps,
                    eta_min=0  # You can set this to a fraction of the initial LR
                ),
                'interval': 'step',
                'frequency': 1
            }

            # Warm-up scheduler
            if self.hparams.warmup_steps > 0:
                def lr_lambda(current_step):
                    if current_step < self.hparams.warmup_steps:
                        return float(current_step) / float(max(1, self.hparams.warmup_steps))
                    else:
                        return 1.0  # The CosineAnnealingLR will handle the rest

                warmup_scheduler = {
                    'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                    'interval': 'step',
                    'frequency': 1
                }
                return [optimizer], [warmup_scheduler, scheduler]
            else:
                return [optimizer], [scheduler]
            
        else:
            return optimizer

    
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #     # Clip gradients to prevent exploding gradients
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #     optimizer.step(closure=optimizer_closure)