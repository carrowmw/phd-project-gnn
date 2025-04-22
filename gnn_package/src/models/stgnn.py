# gnn_package/src/models/stgnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from gnn_package.config import get_config

logger = logging.getLogger(__name__)


class GraphConvolution(nn.Module):
    def __init__(self, config, layer_id, in_features, out_features, bias=True):
        """
        Initialize the GraphConvolution layer with explicit parameters.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object containing global settings
        layer_id : int
            Identifier for this layer
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        bias : bool
            Whether to include bias term (this can remain a default)
        """
        super(GraphConvolution, self).__init__()

        # Store parameters without defaults
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id

        # Get required values from config
        self.use_self_loop = config.model.use_self_loops
        self.normalization = config.model.gcn_normalization
        self.missing_value = config.data.general.missing_value

        # Define learnable parameters
        self.weight = nn.Parameter(
            torch.FloatTensor(self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj, mask=None):
        """
        x: Node features [batch_size, num_nodes, in_features] or [batch_size, in_features]
        adj: Adjacency matrix [num_nodes, num_nodes]
        mask: Mask for valid values [batch_size, num_nodes, 1] or [batch_size, 1]

        Returns:
        --------
        Tensor of shape [batch_size, num_nodes, out_features]
        """
        # Create a binary mask where 1 = valid data, 0 = missing data
        missing_mask = (x != self.missing_value).float()

        # Apply the mask and replace missing values with zeros for computation
        x_masked = x * missing_mask

        # If a separate mask is provided, combine it with the missing mask
        if mask is not None:
            combined_mask = missing_mask * mask
        else:
            combined_mask = missing_mask

        # Check if we're dealing with batched input
        is_batched = len(x.shape) == 3

        if is_batched:
            batch_size, num_nodes, in_features = x.shape
        else:
            num_nodes, in_features = x.shape

        # Check that input features match weight dimensions
        if in_features != self.in_features:
            raise ValueError(
                f"Input features ({in_features}) don't match weight dimensions ({self.in_features})"
            )

        # Check that adjacency matrix dimensions match num_nodes
        if adj.shape[0] != num_nodes:
            raise ValueError(
                f"Adjacency matrix dimension ({adj.shape[0]}) doesn't match number of nodes ({num_nodes})"
            )

        # Add identity to allow self-loops
        adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)

        # Normalize adjacency matrix
        rowsum = adj_with_self.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_adj = torch.matmul(
            torch.matmul(d_mat_inv_sqrt, adj_with_self), d_mat_inv_sqrt
        )

        # Transform node features differently depending on whether we have batched input
        if is_batched:
            # Handle batched data - need to process each batch separately
            outputs = []

            for b in range(batch_size):
                # Extract features for this batch
                batch_features = x_masked[b]  # [num_nodes, in_features]

                # Transform node features
                batch_support = torch.matmul(
                    batch_features, self.weight
                )  # [num_nodes, out_features]

                # Propagate using normalized adjacency
                batch_output = torch.matmul(
                    normalized_adj, batch_support
                )  # [num_nodes, out_features]

                # Add to outputs
                outputs.append(batch_output)

            # Stack back to batched tensor
            output = torch.stack(
                outputs, dim=0
            )  # [batch_size, num_nodes, out_features]

            # Re-apply mask
            if mask is not None:
                output = output * combined_mask
        else:
            # Transform node features
            support = torch.matmul(x_masked, self.weight)  # [num_nodes, out_features]

            # Propagate using normalized adjacency
            output = torch.matmul(normalized_adj, support)  # [num_nodes, out_features]

            # Re-apply mask
            if mask is not None:
                output = output * combined_mask

        # Add bias if needed
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AttentionLayer(nn.Module):
    """
    Attention layer to focus on most relevant nodes and timestamps.
    """

    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        x: Input tensor [batch_size, seq_len/num_nodes, features]
        mask: Binary mask [batch_size, seq_len/num_nodes, 1]
        """
        # Calculate attention scores
        attention_scores = self.attention(x)  # [batch_size, seq_len/num_nodes, 1]

        # Apply mask if provided (set scores to a large negative value)
        if mask is not None:
            # Convert -1 values to mask
            if len(mask.shape) == len(x.shape):
                mask = (mask != -1).float() * (x != -1).float()
            else:
                mask = (x != -1).float()

            # Set masked positions to large negative value
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention to input
        context = torch.sum(x * attention_weights, dim=1)

        return context, attention_weights


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network with attention for missing data.
    """

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        """
        Initialize the TemporalGCN with explicit parameters.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Output feature dimension
        """
        super(TemporalGCN, self).__init__()

        # Store parameters without defaults
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Get values from config
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        num_gc_layers = config.model.num_gc_layers  # This should be required in config

        # Graph Convolutional layers
        self.gc_layers = nn.ModuleList()

        # First layer (input_dim to hidden_dim)
        self.gc_layers.append(
            GraphConvolution(
                config=config,
                layer_id=0,
                in_features=self.input_dim,
                out_features=self.hidden_dim,
            )
        )

        # Additional layers (hidden_dim to hidden_dim)
        for i in range(1, num_gc_layers):
            self.gc_layers.append(
                GraphConvolution(
                    config=config,
                    layer_id=i,
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                )
            )

        # Attention layers
        self.node_attention = AttentionLayer(hidden_dim)
        self.temporal_attention = AttentionLayer(hidden_dim)

        # Recurrent layer for temporal patterns
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask=None):
        """
        x: Node features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        mask: Mask for valid values [batch_size, num_nodes, seq_len, input_dim] or [batch_size, num_nodes, seq_len]
        """
        batch_size, num_nodes, seq_len, features = x.size()

        # Handle mask dimensions
        if mask is not None:
            # If mask has 3 dimensions [batch, nodes, seq_len], expand to 4
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)  # Add feature dimension

                # If we need to match multiple features
                if features > 1:
                    mask = mask.expand(-1, -1, -1, features)

        # Process each time step through the GCN layers
        outputs = []

        for t in range(seq_len):
            # Get features at this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, features]

            # Create mask for this timestep
            if mask is not None:
                mask_t = mask[:, :, t, :]  # [batch_size, num_nodes, features]
            else:
                mask_t = None

            # Apply GCN layers
            for gc_layer in self.gc_layers:
                x_t = gc_layer(x_t, adj, mask_t)
                x_t = F.relu(x_t)  # Activation
                x_t = self.dropout(x_t)  # Apply dropout

            # Store the processed features for this timestep
            outputs.append(x_t)  # [batch_size, num_nodes, hidden_dim]

        # Stack outputs along time dimension
        # This gives us [batch_size, num_nodes, seq_len, hidden_dim]
        temporal_features = torch.stack(outputs, dim=2)

        # Process each node's temporal sequence with GRU
        node_outputs = []

        for n in range(num_nodes):
            # Get temporal data for this node across all batches
            # Shape: [batch_size, seq_len, hidden_dim]
            node_temporal_data = temporal_features[:, n, :, :]

            # Pass through GRU
            # Output shape: [batch_size, seq_len, hidden_dim]
            node_gru_out, _ = self.gru(node_temporal_data)

            # Add to collected outputs
            node_outputs.append(node_gru_out)

        # Stack back to full tensor
        # Shape: [batch_size, num_nodes, seq_len, hidden_dim]
        gru_output = torch.stack(node_outputs, dim=1)

        # Apply final FC layer for output
        # Shape: [batch_size, num_nodes, seq_len, output_dim]
        out = self.fc_out(gru_output)

        # Apply mask if provided to ensure missing values stay missing
        if mask is not None:
            # Ensure mask has right shape
            if mask.shape[-1] == 1 and out.shape[-1] > 1:
                # Expand last dimension if needed
                mask = mask.expand(-1, -1, -1, out.shape[-1])

            # Apply mask
            out = out * mask

        return out


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network with attention for traffic prediction
    """

    def __init__(self, config):
        """
        Initialize STGNN model with configuration.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object
        """
        super(STGNN, self).__init__()

        # Get required parameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        horizon = config.data.general.horizon
        dropout = config.model.dropout
        decoder_layers = (
            config.model.decoder_layers
        )  # This should be required in config

        self.horizon = horizon

        # Encoder: process historical data
        self.encoder = TemporalGCN(
            config=config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        if decoder_layers == 1:
            # If only one layer, use a simple linear layer
            self.decoder = nn.Linear(hidden_dim, output_dim * horizon)
        else:
            decoder_layers_list = []
            decoder_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers_list.append(nn.ReLU())
            decoder_layers_list.append(nn.Dropout(dropout))
            decoder_layers_list.append(nn.Linear(hidden_dim, output_dim * horizon))
            self.decoder = nn.Sequential(*decoder_layers_list)

    def forward(self, x, adj, x_mask=None):
        """
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        # Check for proper shape
        assert len(x.shape) == 4, f"Expected 4D input but got shape {x.shape}"

        batch_size, num_nodes, seq_len, _ = x.size()

        # Encode the input sequence
        # Output shape: [batch_size, num_nodes, seq_len, hidden_dim]
        encoded = self.encoder(x, adj, x_mask)

        # Use the last time step for each node to predict future
        # Shape: [batch_size, num_nodes, hidden_dim]
        last_hidden = encoded[:, :, -1, :]

        # Predict future values
        # Shape: [batch_size, num_nodes, output_dim * horizon]
        future_flat = self.decoder(last_hidden)

        # Reshape to separate time steps
        # Shape: [batch_size, num_nodes, horizon, output_dim]
        predictions = future_flat.reshape(batch_size, num_nodes, self.horizon, -1)

        return predictions


class STGNNTrainer:
    def __init__(self, model, config):
        """
        Initialize the trainer with the model and config.

        Parameters:
        -----------
        model : STGNN
            The model to train
        config : ExperimentConfig
            Configuration object
        """
        # Get device from config or auto-detect
        device_name = getattr(config.training, "device", None)
        if device_name:
            device = torch.device(device_name)
        else:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        logger.info(f"Using device: {device}")

        # Create optimizer based on config
        learning_rate = config.training.learning_rate
        weight_decay = config.training.weight_decay

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Use MSE loss
        criterion = torch.nn.MSELoss(reduction="none")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        logger.info(
            f"STGNNTrainer.__init__(): Model inistialized with {sum(p.numel() for p in model.parameters())} parameters"
        )
        logger.info(f"STGNNTrainer.__init__(): Optimizer: {optimizer}")
        logger.info(f"STGNNTrainer.__init__(): Loss function: {criterion}")

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            x = batch["x"].to(self.device)
            x_mask = batch["x_mask"].to(self.device)
            y = batch["y"].to(self.device)
            y_mask = batch["y_mask"].to(self.device)
            adj = batch["adj"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x, adj, x_mask)

            # Compute loss on valid points only
            loss = self.criterion(y_pred, y)
            if y_mask is not None:
                # Count non-zero elements in mask
                mask_sum = y_mask.sum()
                if mask_sum > 0:
                    loss = (loss * y_mask).sum() / mask_sum
                else:
                    loss = torch.tensor(0.0, device=self.device)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def evaluate(self, dataloader):
        """Evaluate the model on a validation or test set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                x = batch["x"].to(self.device)
                x_mask = batch["x_mask"].to(self.device)
                y = batch["y"].to(self.device)
                y_mask = batch["y_mask"].to(self.device)
                adj = batch["adj"].to(self.device)

                # Forward pass
                y_pred = self.model(x, adj, x_mask)

                # Compute loss on valid points only
                loss = self.criterion(y_pred, y)
                if y_mask is not None:
                    # Count non-zero elements in mask
                    mask_sum = y_mask.sum()
                    if mask_sum > 0:
                        loss = (loss * y_mask).sum() / mask_sum
                    else:
                        loss = torch.tensor(0.0, device=self.device)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)


def create_stgnn_model(config):
    """
    Create a Spatio-Temporal GNN model with parameters from configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    STGNN
        Configured model instance
    """
    # Validate that all required configuration parameters are present
    try:
        # This will raise an error if any required parameter is missing
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise

    # Log the configuration being used
    logger.info(f"Creating STGNN model with config from {config.config_path}")
    logger.info(f"  input_dim: {config.model.input_dim}")
    logger.info(f"  hidden_dim: {config.model.hidden_dim}")
    logger.info(f"  output_dim: {config.model.output_dim}")
    logger.info(f"  horizon: {config.data.general.horizon}")
    logger.info(f"  num_layers: {config.model.num_layers}")
    logger.info(f"  num_gc_layers: {config.model.num_gc_layers}")
    logger.info(f"  decoder_layers: {config.model.decoder_layers}")
    logger.info(f"  dropout: {config.model.dropout}")

    # Create model with config
    model = STGNN(config=config)

    return model
