# gnn_package/src/models/stgnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [num_nodes, num_nodes]
        """
        # First transform node features
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]

        # Then propagate using normalized adjacency matrix
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

        # Propagate node features using normalized adjacency
        output = torch.matmul(normalized_adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network layer that combines
    graph convolutions with GRU for temporal dynamics
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(TemporalGCN, self).__init__()

        # Graph Convolutional layers
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)

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
        mask: Mask for valid values [batch_size, num_nodes, seq_len]
        """
        batch_size, num_nodes, seq_len, feat_dim = x.size()

        # Process each time step with GCN
        gcn_outputs = []
        for t in range(seq_len):
            # Get features at current time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, input_dim]

            # Apply graph convolution
            h = F.relu(self.gc1(x_t, adj))
            h = self.dropout(h)
            h = F.relu(self.gc2(h, adj))
            h = self.dropout(h)

            gcn_outputs.append(h)

        # Stack GCN outputs along time dimension
        gcn_out = torch.stack(
            gcn_outputs, dim=2
        )  # [batch_size, num_nodes, seq_len, hidden_dim]

        # Reshape for RNN: combine batch and nodes
        rnn_in = gcn_out.reshape(batch_size * num_nodes, seq_len, -1)

        # Apply RNN
        rnn_out, _ = self.gru(rnn_in)

        # Reshape back
        rnn_out = rnn_out.reshape(batch_size, num_nodes, seq_len, -1)

        # Apply output layer
        out = self.fc_out(rnn_out)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Add feature dimension
            out = out * mask  # Zero out invalid positions

        return out


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for traffic prediction
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, horizon, num_layers=1, dropout=0.2
    ):
        super(STGNN, self).__init__()

        self.horizon = horizon

        # Encoder: process historical data
        self.encoder = TemporalGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Decoder: predict future values
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * horizon),
        )

    def forward(self, x, adj, x_mask=None):
        """
        x: Input features [batch_size, num_nodes, input_seq_len, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        x_mask: Mask for input [batch_size, num_nodes, input_seq_len]

        Returns:
        Predictions [batch_size, num_nodes, horizon, output_dim]
        """
        batch_size, num_nodes, seq_len, _ = x.size()

        # Encode the input sequence
        encoded = self.encoder(x, adj, x_mask)

        # Use the last time step for each node to predict future
        last_hidden = encoded[:, :, -1, :]  # [batch_size, num_nodes, hidden_dim]

        # Predict future values
        future_flat = self.decoder(
            last_hidden
        )  # [batch_size, num_nodes, output_dim * horizon]

        # Reshape to separate time steps
        predictions = future_flat.reshape(batch_size, num_nodes, self.horizon, -1)

        return predictions


class STGNNTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

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
                loss = (loss * y_mask).sum() / y_mask.sum()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader):
        """Evaluate model on a dataset"""
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
                    loss = (loss * y_mask).sum() / y_mask.sum()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches


# Example usage
def create_stgnn_model(
    input_dim=1, hidden_dim=64, output_dim=1, horizon=6, num_layers=2
):
    """Create a Spatio-Temporal GNN model with specified parameters"""
    model = STGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        horizon=horizon,
        num_layers=num_layers,
    )
    return model
