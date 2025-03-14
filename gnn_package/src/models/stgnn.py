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
        # print(f"DEBUG: GraphConvolution.forward - x shape: {x.shape}")
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
        # print(f"DEBUG: GraphConvolution.forward - output shape: {output.shape}")

        # Add bias if needed)
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
        batch_size, num_nodes, seq_len, features = x.size()

        # Process each time step through the GCN layers
        outputs = []
        for t in range(seq_len):
            # Get features at this time step
            x_t = x[:, :, t, :]  # [batch_size, num_nodes, features]

            # Apply GC layers
            h = self.gc1(x_t, adj)  # First GC layer
            h = F.relu(h)  # Activation
            h = self.dropout(h)  # Apply dropout
            h = self.gc2(h, adj)  # Second GC layer

            outputs.append(h)

        # Stack outputs along time dimension
        out_stacked = torch.stack(
            outputs, dim=2
        )  # [batch_size, num_nodes, seq_len, hidden_dim]

        # Reshape for GRU: [batch_size * num_nodes, seq_len, hidden_dim]
        out_gru = out_stacked.view(batch_size * num_nodes, seq_len, -1)

        # Apply GRU for temporal modeling
        out_gru, _ = self.gru(out_gru)

        # Reshape back: [batch_size, num_nodes, seq_len, hidden_dim]
        out_reshaped = out_gru.view(batch_size, num_nodes, seq_len, -1)

        # Apply final FC layer for each time step
        out_final = self.fc_out(out_reshaped)

        # Apply mask if provided
        if mask is not None:
            # Ensure mask has right shape
            if len(mask.shape) == 3:  # [batch, nodes, seq_len]
                mask = mask.unsqueeze(-1)  # Add feature dimension

            # Expand mask if needed
            if mask.shape[3] == 1 and out_final.shape[3] > 1:
                mask = mask.expand(-1, -1, -1, out_final.shape[3])

            # Apply mask
            out_final = out_final * mask

        return out_final


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
        print(f"DEBUG: STGNN.forward - input shape: {x.shape}")

        # Enocde the input sequence
        encoded = self.encoder(x, adj, x_mask)
        print(f"STGNN.forward - encoded shape: {encoded.shape}")

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

            print(
                f"DEBUG: STGNNTrainer.train_epoch before model forward - x shape: {x.shape}, x_mask shape: {x_mask.shape}"
            )

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x, adj, x_mask)

            print(
                f"DEBUG: STGNNTrainer.train_epoch after model forward - y_pred shape: {y_pred.shape}, y shape: {y.shape}"
            )

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
