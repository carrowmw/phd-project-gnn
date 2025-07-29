import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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

class GraphAttentionHead(nn.Module):
    """
    Single attention head for Graph Attention Networks (GAT).
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionHead, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # Learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # Initialize weights using Glorot initialization
        nn.init.xavier_uniform_(self.W)

        # Attention parameters (learnable)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a)

        # Leaky ReLU activation for attention scores
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj, mask=None):
        """
        Forward pass for a single attention head.

        Parameters:
        -----------
        x : torch.Tensor
            Node features [batch_size, num_nodes, in_features]
        adj : torch.Tensor
            Adjacency matrix [num_nodes, num_nodes]
        mask : torch.Tensor, optional
            Mask for valid node features [batch_size, num_nodes, 1]

        Returns:
        --------
        torch.Tensor
            Attention-weighted node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation of input features
        h = torch.matmul(x, self.W)  # [batch_size, num_nodes, out_features]

        # Calculate attention coefficients for each edge
        # First, prepare concatenated features for all node pairs
        a_input = torch.cat([
            h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, self.out_features),
            h.repeat(1, num_nodes, 1)
        ], dim=2).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)

        # Apply attention mechanism
        e = self.leaky_relu(torch.matmul(a_input, self.a)).squeeze(-1)  # [batch_size, num_nodes, num_nodes]

        # Mask out non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_nodes, num_nodes]
        attention = torch.where(adj > 0, e, zero_vec)

        # Apply softmax to get normalized attention coefficients
        attention = F.softmax(attention, dim=2)

        # Apply dropout to attention coefficients
        attention = self.dropout_layer(attention)

        # Apply masked attention to compute weighted sum of neighbor features
        h_prime = torch.bmm(attention, h)  # [batch_size, num_nodes, out_features]

        # Apply mask if provided
        if mask is not None:
            h_prime = h_prime * mask

        return h_prime

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(
                GraphAttentionHead(in_features, out_features // num_heads, dropout, alpha)
            )

        # Output feature size is divided among heads
        self.out_features = out_features

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask=None):
        # x: [batch_size, num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        # mask: [batch_size, num_nodes, 1]

        # Apply each attention head
        head_outputs = [attention(x, adj, mask) for attention in self.attentions]

        # Concatenate heads (or average them if specified)
        x = torch.cat(head_outputs, dim=-1)

        # Apply final dropout
        x = self.dropout(x)

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for sequence data.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for query, key, value
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, seq_len, hidden_dim]
        mask : torch.Tensor, optional
            Attention mask [batch_size, seq_len, seq_len]

        Returns:
        --------
        torch.Tensor
            Attention output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            if len(mask.shape) == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

            # Apply mask (set masked positions to -inf)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)

        # Transpose and reshape back to original shape
        # [batch_size, seq_len, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Apply output projection
        output = self.output_proj(context)

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


class TemporalProcessor(nn.Module):
    """Hybrid model combining GRU and attention for temporal processing."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, attention_heads=4, dropout=0.2):
        super(TemporalProcessor, self).__init__()

        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Temporal attention for capturing non-sequential dependencies
        self.temporal_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout
        )

        # Fusion layer to combine GRU and attention outputs
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch_size * num_nodes, seq_len, hidden_dim]
        # mask: [batch_size * num_nodes, seq_len, 1]

        # Process with GRU
        gru_output, _ = self.gru(x)

        # Process with temporal attention
        if mask is not None:
            # Create attention mask where valid positions influence each other
            attn_mask = mask @ mask.transpose(-2, -1)  # [batch*nodes, seq, seq]
        else:
            attn_mask = None

        attn_output = self.temporal_attention(x, mask=attn_mask)

        # Combine outputs
        combined = torch.cat([gru_output, attn_output], dim=-1)
        output = self.fusion(combined)
        output = self.dropout(output)

        # Apply mask if provided
        if mask is not None:
            output = output * mask

        return output

class EnhancedDecoder(nn.Module):
    """
    Decoder with multi-scale processing for different prediction horizons.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        horizon,
        attention_heads=4,
        dropout=0.2
    ):
        super(EnhancedDecoder, self).__init__()

        self.horizon = horizon

        # Attention mechanism for input context
        self.context_attention = MultiHeadAttention(
            hidden_dim=input_dim,
            num_heads=attention_heads,
            dropout=dropout
        )

        # Horizon-specific projection layers
        # Different weights for different prediction horizons
        self.horizon_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # Add softplus for positive outputs
            ) for _ in range(horizon)
        ])

    def forward(self, encoder_output, encoder_mask=None):
        # encoder_output: [batch_size, num_nodes, seq_len, input_dim]
        # encoder_mask: [batch_size, num_nodes, seq_len, 1]

        batch_size, num_nodes, seq_len, input_dim = encoder_output.shape

        # Get context-aware representations
        # Reshape for attention: [batch*nodes, seq, dim]
        flat_encoder = encoder_output.view(batch_size * num_nodes, seq_len, input_dim)
        flat_mask = encoder_mask.view(batch_size * num_nodes, seq_len, 1) if encoder_mask is not None else None

        # Apply context attention
        context = self.context_attention(flat_encoder, mask=flat_mask)

        # Last hidden state as input to decoder
        last_hidden = context[:, -1, :].unsqueeze(1)  # [batch*nodes, 1, dim]

        # Generate predictions for each horizon step with different projections
        predictions = []
        for h in range(self.horizon):
            # Apply horizon-specific projection
            pred = self.horizon_projections[h](last_hidden)
            predictions.append(pred)

        # Stack predictions along horizon dimension
        # [batch*nodes, horizon, output_dim]
        stacked_preds = torch.cat(predictions, dim=1)

        # Reshape back to [batch, nodes, horizon, output_dim]
        output = stacked_preds.view(batch_size, num_nodes, self.horizon, -1)

        return output