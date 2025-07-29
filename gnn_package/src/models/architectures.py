# gnn_package/src/models/stgnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, MultiHeadAttention, TemporalProcessor, EnhancedDecoder, GraphAttentionLayer

class STGNN(nn.Module):
    """
    Original Spatio-Temporal Graph Neural Network with GraphConvolution and GRU.
    Refactored to use the new layer naming conventions for consistency.
    """
    def __init__(self, config):
        super(STGNN, self).__init__()

        # Get required parameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        horizon = config.data.general.horizon
        dropout = config.model.dropout
        num_layers = config.model.num_layers
        num_gc_layers = config.model.num_gc_layers

        # Store horizon for output generation
        self.horizon = horizon

        # Encoder: using GraphConvolution layers
        self.gc_layers = nn.ModuleList()

        # First GC layer (input_dim to hidden_dim)
        self.gc_layers.append(
            GraphConvolution(
                config=config,
                layer_id=0,
                in_features=input_dim,
                out_features=hidden_dim,
            )
        )

        # Additional GC layers (hidden_dim to hidden_dim)
        for i in range(1, num_gc_layers):
            self.gc_layers.append(
                GraphConvolution(
                    config=config,
                    layer_id=i,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )

        # Attention for node interaction
        self.node_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=config.model.attention_heads,
            dropout=dropout,
        )

        # Recurrent layer for temporal patterns
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.fc_out = nn.Sequential(
    nn.Linear(hidden_dim, output_dim),
    nn.Softplus()  # Ensures positive outputs
)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, x_mask=None):
        """
        Forward pass for STGNN with encoder-decoder architecture.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        --------
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        # Check for proper shape
        assert len(x.shape) == 4, f"Expected 4D input but got shape {x.shape}"

        batch_size, num_nodes, seq_len, _ = x.size()

        # Process through graph convolution layers at each time step
        spatial_features = []

        for t in range(seq_len):
            # Get features for this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, input_dim]

            # Get mask for this time step if available
            mask_t = x_mask[:, :, t, :] if x_mask is not None else None

            # Process through GC layers
            h = x_t
            for gc_layer in self.gc_layers:
                h = gc_layer(h, adj, mask_t)
                h = F.relu(h)
                h = self.dropout(h)

            spatial_features.append(h)

        # Stack spatial features across time
        # [batch_size, num_nodes, seq_len, hidden_dim]
        encoder_outputs = torch.stack(spatial_features, dim=2)

        # Apply node attention to capture inter-node dependencies
        # First reshape for attention
        # [batch_size * seq_len, num_nodes, hidden_dim]
        reshaped_outputs = encoder_outputs.permute(0, 2, 1, 3).contiguous()
        reshaped_outputs = reshaped_outputs.view(batch_size * seq_len, num_nodes, -1)

        # Apply node attention
        if x_mask is not None:
            # Reshape mask too
            reshaped_mask = x_mask.permute(0, 2, 1, 3).contiguous()
            reshaped_mask = reshaped_mask.view(batch_size * seq_len, num_nodes, -1)

            # Create attention mask
            attention_mask = torch.bmm(reshaped_mask, reshaped_mask.transpose(-2, -1))
        else:
            attention_mask = None

        # Apply attention
        node_context = self.node_attention(reshaped_outputs)

        # Reshape back
        # [batch_size, seq_len, num_nodes, hidden_dim]
        node_context = node_context.view(batch_size, seq_len, num_nodes, -1)
        # [batch_size, num_nodes, seq_len, hidden_dim]
        node_context = node_context.permute(0, 2, 1, 3).contiguous()

        # Process with GRU for each node
        # [batch_size * num_nodes, seq_len, hidden_dim]
        gru_input = node_context.view(batch_size * num_nodes, seq_len, -1)

        # Pass through GRU
        gru_output, _ = self.gru(gru_input)

        # Use the last hidden state to make predictions
        # [batch_size * num_nodes, hidden_dim]
        last_hidden = gru_output[:, -1, :]


class ImprovedSTGNN(nn.Module):
    """
    Improved Spatio-Temporal Graph Neural Network with attention mechanisms.
    Uses graph attention for spatial processing and hybrid temporal processing.
    """

    def __init__(self, config):
        super(ImprovedSTGNN, self).__init__()

        # Get required parameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        horizon = config.data.general.horizon
        dropout = config.model.dropout
        num_layers = config.model.num_layers
        attention_heads = config.model.attention_heads

        # Spatial processing with graph attention
        self.spatial_layers = nn.ModuleList()

        # Input layer
        self.spatial_layers.append(
            GraphAttentionLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.spatial_layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=attention_heads,
                    dropout=dropout
                )
            )

        # Temporal processing
        self.temporal_processor = TemporalProcessor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            attention_heads=attention_heads,
            dropout=dropout
        )

        # Decoder
        self.decoder = EnhancedDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            attention_heads=attention_heads,
            dropout=dropout
        )

    def forward(self, x, adj, x_mask=None):
        """
        Forward pass for improved STGNN.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        --------
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        batch_size, num_nodes, seq_len, input_dim = x.shape

        # Process each time step with spatial layers
        spatial_features = []

        for t in range(seq_len):
            # Get features for this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, input_dim]

            # Get mask for this time step if available
            mask_t = x_mask[:, :, t, :] if x_mask is not None else None

            # Process through spatial layers
            h = x_t
            for layer in self.spatial_layers:
                h = layer(h, adj, mask_t)
                h = F.relu(h)

            spatial_features.append(h)

        # Stack spatial features to form spatio-temporal tensor
        # [batch_size, num_nodes, seq_len, hidden_dim]
        spatial_output = torch.stack(spatial_features, dim=2)

        # Flatten for temporal processing
        # [batch_size * num_nodes, seq_len, hidden_dim]
        temporal_input = spatial_output.view(batch_size * num_nodes, seq_len, -1)

        # Flatten mask if provided
        if x_mask is not None:
            temporal_mask = x_mask.view(batch_size * num_nodes, seq_len, -1)
        else:
            temporal_mask = None

        # Process with temporal module
        temporal_output = self.temporal_processor(temporal_input, temporal_mask)

        # Reshape for decoder
        # [batch_size, num_nodes, seq_len, hidden_dim]
        decoder_input = temporal_output.view(batch_size, num_nodes, seq_len, -1)

        # Generate predictions
        predictions = self.decoder(decoder_input, x_mask)

        return predictions

class GATWithGRU(nn.Module):
    """
    Hybrid architecture that uses GAT for spatial processing but keeps simple GRU for temporal.
    This maintains the benefits of graph attention while using the proven GRU temporal model.
    """

    def __init__(self, config):
        super(GATWithGRU, self).__init__()

        # Get required parameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        horizon = config.data.general.horizon
        dropout = config.model.dropout
        num_layers = config.model.num_layers
        attention_heads = config.model.attention_heads

        # Store horizon for predictions
        self.horizon = horizon

        # Spatial processing with graph attention
        self.spatial_layers = nn.ModuleList()

        # Input layer
        self.spatial_layers.append(
            GraphAttentionLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
        )

        # Additional layers if needed
        for _ in range(num_layers - 1):
            self.spatial_layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=attention_heads,
                    dropout=dropout
                )
            )

        # Standard GRU for temporal processing
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection for each horizon step
        self.horizon_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # Ensures positive outputs
            ) for _ in range(horizon)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, x_mask=None):
        """
        Forward pass for GATWithGRU.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        --------
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        batch_size, num_nodes, seq_len, _ = x.shape

        # Process each time step with spatial layers
        spatial_features = []

        for t in range(seq_len):
            # Get features for this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, input_dim]

            # Get mask for this time step if available
            mask_t = x_mask[:, :, t, :] if x_mask is not None else None

            # Process through spatial layers
            h = x_t
            for layer in self.spatial_layers:
                h = layer(h, adj, mask_t)
                h = F.relu(h)
                h = self.dropout(h)

            spatial_features.append(h)

        # Stack spatial features to form spatio-temporal tensor
        # [batch_size, num_nodes, seq_len, hidden_dim]
        spatial_output = torch.stack(spatial_features, dim=2)

        # Reshape for GRU processing
        # [batch_size * num_nodes, seq_len, hidden_dim]
        gru_input = spatial_output.view(batch_size * num_nodes, seq_len, -1)

        # Process with GRU
        gru_output, _ = self.gru(gru_input)

        # Get the last hidden state for predictions
        # [batch_size * num_nodes, hidden_dim]
        last_hidden = gru_output[:, -1, :]

        # Generate predictions for each horizon step with separate projections
        predictions = []
        for h in range(self.horizon):
            step_pred = self.horizon_projections[h](last_hidden)
            predictions.append(step_pred.unsqueeze(1))

        # Concatenate predictions along horizon dimension
        # [batch_size * num_nodes, horizon, output_dim]
        stacked_preds = torch.cat(predictions, dim=1)

        # Reshape to [batch_size, num_nodes, horizon, output_dim]
        output = stacked_preds.view(batch_size, num_nodes, self.horizon, -1)

        return output

class FullAttentionSTGNN(nn.Module):
    """
    Fully attention-based Spatio-Temporal Graph Neural Network without GRU components.
    Uses attention for both spatial and temporal dimensions.
    """

    def __init__(self, config):
        super(FullAttentionSTGNN, self).__init__()

        # Get required parameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        horizon = config.data.general.horizon
        dropout = config.model.dropout
        num_layers = config.model.num_layers
        attention_heads = config.model.attention_heads

        # Spatial attention layers
        self.spatial_attention_layers = nn.ModuleList()

        # First layer converts from input_dim to hidden_dim
        self.spatial_attention_layers.append(
            GraphAttentionLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
        )

        # Additional spatial attention layers
        for _ in range(num_layers - 1):
            self.spatial_attention_layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=attention_heads,
                    dropout=dropout
                )
            )

        # Temporal attention for sequence modeling
        self.temporal_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout
        )

        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network for temporal processing
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Decoder attention for generating future predictions
        self.decoder_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout
        )

        # Output projections for each horizon step
        self.horizon_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # Ensures positive outputs
            ) for _ in range(horizon)
        ])

        # Horizon
        self.horizon = horizon

        # Final dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, x_mask=None):
        """
        Forward pass for FullAttentionSTGNN.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        --------
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        batch_size, num_nodes, seq_len, _ = x.shape

        # Process each time step with spatial attention
        spatial_features = []

        for t in range(seq_len):
            # Get features for this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, input_dim]

            # Get mask for this time step if available
            mask_t = x_mask[:, :, t, :] if x_mask is not None else None

            # Process through spatial attention layers
            h = x_t
            for layer in self.spatial_attention_layers:
                h = layer(h, adj, mask_t)
                h = self.dropout(h)

            spatial_features.append(h)

        # Stack spatial features to form spatio-temporal tensor
        # [batch_size, num_nodes, seq_len, hidden_dim]
        spatial_output = torch.stack(spatial_features, dim=2)

        # Process each node's temporal sequence with temporal attention
        temporal_features = []

        for n in range(num_nodes):
            # Get features for this node
            node_features = spatial_output[:, n, :, :]  # [batch_size, seq_len, hidden_dim]

            # Get mask for this node if available
            if x_mask is not None:
                node_mask = x_mask[:, n, :, 0]  # [batch_size, seq_len]
                # Create attention mask
                attn_mask = node_mask.unsqueeze(-1) @ node_mask.unsqueeze(-2)  # [batch_size, seq_len, seq_len]
            else:
                attn_mask = None

            # Apply temporal attention (with residual connection)
            attn_output = self.temporal_attention(node_features, attn_mask)
            node_features = self.layer_norm1(node_features + attn_output)

            # Apply feed-forward network (with residual connection)
            ff_output = self.feed_forward(node_features)
            node_features = self.layer_norm2(node_features + ff_output)

            temporal_features.append(node_features)

        # Stack to get [batch_size, num_nodes, seq_len, hidden_dim]
        temporal_output = torch.stack(temporal_features, dim=1)

        # Use the last sequence position as context for prediction
        context = temporal_output[:, :, -1, :]  # [batch_size, num_nodes, hidden_dim]

        # Generate predictions for each horizon step
        predictions = []

        for h in range(self.horizon):
            # Use the same context for each horizon step
            # But apply different projection
            step_pred = self.horizon_projections[h](context)
            predictions.append(step_pred.unsqueeze(2))  # Add horizon dimension

        # Concatenate along horizon dimension
        # [batch_size, num_nodes, horizon, output_dim]
        output = torch.cat(predictions, dim=2)

        return output