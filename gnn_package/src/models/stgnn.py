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


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for capturing complex dependencies across nodes or time steps.

    This implementation follows the attention mechanism from "Attention Is All You Need"
    but is adapted for spatio-temporal graph data.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Initialize the multi-head attention module.

        Parameters:
        -----------
        hidden_dim : int
            Dimension of input and output features
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        # Ensure hidden_dim is divisible by num_heads for proper splitting
        assert (
            hidden_dim % num_heads == 0
        ), "Hidden dimension must be divisible by number of heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for query, key, and value
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out = nn.Linear(hidden_dim, hidden_dim)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass for multi-head attention.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [batch_size, seq_len, hidden_dim]
            or [batch_size, num_nodes, hidden_dim]
        mask : torch.Tensor, optional
            Mask tensor with shape [batch_size, seq_len, seq_len]
            or [batch_size, num_nodes, num_nodes]
            1 indicates valid positions, 0 indicates positions to mask
        return_attention : bool
            Whether to return attention weights for visualization/analysis

        Returns:
        --------
        torch.Tensor
            Output tensor with shape [batch_size, seq_len, hidden_dim]
            or [batch_size, num_nodes, hidden_dim]
        torch.Tensor, optional
            Attention weights if return_attention is True
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
        queries = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim)
        keys = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim)
        values = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        # [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len]
        # -> [batch_size, num_heads, seq_len, seq_len]
        energy = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention if needed
            if len(mask.shape) == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

            # Apply mask (using -1e9 for numerical stability)
            energy = energy.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        # Apply dropout to attention weights
        attention = self.dropout(attention)

        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        out = torch.matmul(attention, values)

        # Reshape and concatenate heads
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        out = out.transpose(1, 2)
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, hidden_dim]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        # Apply output projection
        out = self.out(out)

        if return_attention:
            # Return both output and attention weights for visualization
            return out, attention

        return out


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
        Vectorized implementation of temporal GCN processing.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        mask: Mask for valid values [batch_size, num_nodes, seq_len, input_dim]

        Returns:
        --------
        Output features [batch_size, num_nodes, seq_len, hidden_dim]
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

        # Reshape to process all time steps at once
        # [batch_size, num_nodes, seq_len, features] -> [batch_size * seq_len, num_nodes, features]
        x_reshaped = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * seq_len, num_nodes, features)
        )

        # If mask exists, reshape it the same way
        if mask is not None:
            mask_reshaped = (
                mask.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size * seq_len, num_nodes, features)
            )
        else:
            mask_reshaped = None

        # Process through GCN layers
        h = x_reshaped
        for gc_layer in self.gc_layers:
            h = gc_layer(h, adj, mask_reshaped)
            h = F.relu(h)  # Activation
            h = self.dropout(h)  # Apply dropout

        # Reshape back to original dimensions
        # [batch_size * seq_len, num_nodes, hidden_dim] -> [batch_size, seq_len, num_nodes, hidden_dim]
        h = h.view(batch_size, seq_len, num_nodes, self.hidden_dim).permute(0, 2, 1, 3)

        # Process each node's temporal sequence with GRU
        # Reshape for GRU: [batch_size, num_nodes, seq_len, hidden_dim] -> [batch_size * num_nodes, seq_len, hidden_dim]
        gru_input = h.contiguous().view(
            batch_size * num_nodes, seq_len, self.hidden_dim
        )

        # Pass through GRU
        gru_output, _ = self.gru(gru_input)

        # Reshape back: [batch_size * num_nodes, seq_len, hidden_dim] -> [batch_size, num_nodes, seq_len, hidden_dim]
        gru_output = gru_output.view(batch_size, num_nodes, seq_len, self.hidden_dim)

        # Apply final FC layer for output
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


class TemporalDecoder(nn.Module):
    """
    Temporal decoder for generating future predictions using an autoregressive approach.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        horizon,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
    ):
        """
        Initialize the temporal decoder.

        Parameters:
        -----------
        input_dim : int
            Dimension of input features (typically the encoder's hidden dimension)
        hidden_dim : int
            Dimension of decoder's hidden state
        output_dim : int
            Dimension of output features
        horizon : int
            Number of future time steps to predict
        num_layers : int
            Number of GRU layers
        dropout : float
            Dropout probability
        use_attention : bool
            Whether to use attention over encoder outputs
        """
        super(TemporalDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.use_attention = use_attention
        self.num_layers = num_layers  # Make sure this is stored

        # GRU for processing sequential data
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Initialize attention mechanism if enabled
        if use_attention:
            # Either use simple attention
            self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
            # Or use multi-head attention if you've implemented it
            # self.attention = MultiHeadAttention(hidden_dim, num_heads=4)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Projection for input
        self.input_proj = nn.Linear(output_dim, input_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_state,
        encoder_outputs=None,
        targets=None,
        teacher_forcing_ratio=0.0,
    ):
        """
        Forward pass for the temporal decoder.

        Parameters:
        -----------
        encoder_state : torch.Tensor
            Final encoder state [batch_size, num_nodes, hidden_dim] or
            [num_layers, batch_size, num_nodes, hidden_dim]
        encoder_outputs : torch.Tensor, optional
            All encoder outputs for attention [batch_size, num_nodes, seq_len, hidden_dim]
        targets : torch.Tensor, optional
            Target values for teacher forcing [batch_size, num_nodes, horizon, output_dim]
        teacher_forcing_ratio : float
            Probability of using teacher forcing (0 = never, 1 = always)

        Returns:
        --------
        torch.Tensor
            Predicted output for future time steps [batch_size, num_nodes, horizon, output_dim]
        """
        # Handle different possible shapes of encoder_state
        if len(encoder_state.shape) == 3:  # [batch_size, num_nodes, hidden_dim]
            batch_size, num_nodes, _ = encoder_state.size()
            # Reshape to match GRU's expected hidden state format
            hidden = encoder_state.view(batch_size * num_nodes, -1)
            hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        elif (
            len(encoder_state.shape) == 4
        ):  # [num_layers, batch_size, num_nodes, hidden_dim]
            num_layers, batch_size, num_nodes, _ = encoder_state.size()
            hidden = encoder_state.view(num_layers, batch_size * num_nodes, -1)
        else:
            raise ValueError(f"Unexpected encoder_state shape: {encoder_state.shape}")

        device = encoder_state.device

        # Initialize output tensor
        outputs = torch.zeros(
            batch_size, num_nodes, self.horizon, self.output_dim, device=device
        )

        # Initialize decoder input (zeros)
        decoder_input = torch.zeros(
            batch_size * num_nodes, 1, self.input_dim, device=device
        )

        # Auto-regressive decoding
        for t in range(self.horizon):
            # GRU forward pass (all nodes at once)
            output, hidden = self.gru(decoder_input, hidden)

            # Apply attention if enabled and encoder outputs are provided
            if (
                self.use_attention
                and encoder_outputs is not None
                and hasattr(self, "attention")
            ):
                # Reshape encoder outputs for attention
                nodes_encoder_outputs = encoder_outputs.view(
                    batch_size * num_nodes, -1, self.hidden_dim
                )

                # How to apply attention depends on the type of attention mechanism
                # Simple attention using linear layer
                if isinstance(self.attention, nn.Linear):
                    # Concatenate current output with each encoder output
                    # This is a simple form of attention - more sophisticated
                    # implementations would use query-key-value calculations
                    seq_len = nodes_encoder_outputs.size(1)
                    output_expanded = output.repeat(1, seq_len, 1)
                    attention_input = torch.cat(
                        [output_expanded, nodes_encoder_outputs], dim=2
                    )
                    attention_weights = torch.softmax(
                        self.attention(attention_input), dim=1
                    )
                    context = torch.sum(
                        attention_weights * nodes_encoder_outputs, dim=1, keepdim=True
                    )
                    # Combine attention context with GRU output
                    output = output + context
                # If using MultiHeadAttention
                else:
                    # Assume self.attention is a module with a compatible interface
                    context = self.attention(output, nodes_encoder_outputs)
                    output = output + context

            # Apply dropout
            output = self.dropout(output)

            # Generate prediction for this time step
            prediction = self.fc_out(
                output.squeeze(1)
            )  # [batch_size * num_nodes, output_dim]

            # Reshape prediction back to [batch_size, num_nodes, output_dim]
            prediction = prediction.view(batch_size, num_nodes, self.output_dim)

            # Store prediction
            outputs[:, :, t, :] = prediction

            # Prepare input for next time step
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth as input
                next_input = self.input_proj(targets[:, :, t, :])
                # Reshape for GRU input
                decoder_input = next_input.view(
                    batch_size * num_nodes, 1, self.input_dim
                )
            else:
                # Autoregressive: use own prediction as input
                next_input = self.input_proj(prediction)
                # Reshape for GRU input
                decoder_input = next_input.view(
                    batch_size * num_nodes, 1, self.input_dim
                )

        return outputs


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network with attention for traffic prediction.

    Updated with vectorized processing, multi-head attention, and explicit decoder.
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
        num_layers = config.model.num_layers
        num_gc_layers = config.model.num_gc_layers
        decoder_layers = config.model.decoder_layers

        # Store parameters
        self.horizon = horizon

        # Encoder: using the improved TemporalGCN
        self.encoder = TemporalGCN(
            config=config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        # Attention for node interaction
        self.node_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=config.model.attention_heads,
            dropout=dropout,
        )

        # Decoder: using the new explicit TemporalDecoder
        self.decoder = TemporalDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            num_layers=decoder_layers,
            dropout=dropout,
            use_attention=True,
        )

    def forward(self, x, adj, x_mask=None, teacher_forcing_ratio=0.0):
        """
        Forward pass for STGNN with encoder-decoder architecture.

        Parameters:
        -----------
        x: Input features [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, seq_len, input_dim]
        teacher_forcing_ratio: Probability of using teacher forcing during training

        Returns:
        --------
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        # Check for proper shape
        assert len(x.shape) == 4, f"Expected 4D input but got shape {x.shape}"

        batch_size, num_nodes, seq_len, _ = x.size()

        # Encode the input sequence using the improved TemporalGCN
        # Output shape: [batch_size, num_nodes, seq_len, hidden_dim]
        encoder_outputs = self.encoder(x, adj, x_mask)

        # Use the last hidden state as the context for each node
        # [batch_size, num_nodes, hidden_dim]
        last_hidden = encoder_outputs[:, :, -1, :]

        # Apply node attention to capture inter-node dependencies
        # [batch_size, num_nodes, hidden_dim]
        node_context = self.node_attention(last_hidden)

        # No need to reshape - pass encoder state directly
        # The decoder will handle the reshaping correctly

        # Generate future predictions using the decoder
        # [batch_size, num_nodes, horizon, output_dim]
        predictions = self.decoder(
            encoder_state=node_context,  # Pass as [batch_size, num_nodes, hidden_dim]
            encoder_outputs=encoder_outputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

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
    logger.info(f"  attention_heads: {config.model.attention_heads}")

    # Create model with config
    model = STGNN(config=config)

    return model
