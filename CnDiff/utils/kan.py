import torch
import torch.nn as nn
import torch.nn.functional as F

from rational_kat_cu.kat_rational import KAT_Group


class KAN(nn.Module):
    """
    A Kolmogorov-Arnold Network (KAN) implementation structured as a two-layer MLP.

    This network takes feature, hidden, and prediction dimensions to create a KAN-based
    block that can serve as a direct replacement for standard MLP layers in models
    like Vision Transformers or MLP-Mixers.

    Args:
        feature_dim (int): The number of input features.
        hidden_dim (int): The number of features in the hidden layer.
        pred_len (int): The number of output features. For use cases like t_phi
            that require shape preservation, this is internally overridden to be
            equal to feature_dim.
        grid_size (int): The number of grid intervals for the spline basis functions.
        spline_order (int): The order of the splines.
        base_activation (nn.Module): A base activation function applied to the input
            before the linear transformation (e.g., nn.GELU).
        grid_range (list): A list of two numbers specifying the range of the grid.
        drop (float): The dropout rate.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        pred_len,
        grid_size=5,
        spline_order=3,
        base_activation=nn.GELU,
        grid_range=[-1, 1],
        drop=0.0,
    ):
        super(KAN, self).__init__()

        # Define the layer structure. For shape-preserving networks like t_phi,
        # the output dimension must match the input dimension.
        # We enforce this by setting the final layer's output to feature_dim.
        self.layers_hidden = [feature_dim, hidden_dim, feature_dim]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()
        self.grid_range = grid_range

        # Initialize lists for KAN components
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        self.prelus = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Loop through the layers to initialize weights, norms, and grids
        for i, (in_features, out_features) in enumerate(
            zip(self.layers_hidden, self.layers_hidden[1:])
        ):
            # Base linear transformation weights
            self.base_weights.append(
                nn.Parameter(torch.randn(out_features, in_features))
            )
            # Spline transformation weights
            self.spline_weights.append(
                nn.Parameter(
                    torch.randn(out_features, in_features, grid_size + spline_order)
                )
            )
            # Layer normalization for stable training
            self.layer_norms.append(nn.LayerNorm(out_features))
            # PReLU for learnable non-linearity
            self.prelus.append(nn.PReLU())
            # Dropout for regularization
            self.dropouts.append(nn.Dropout(drop))

            # Compute and register the grid for spline calculations
            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            grid = (
                torch.linspace(
                    self.grid_range[0] - h * spline_order,
                    self.grid_range[1] + h * spline_order,
                    grid_size + 2 * spline_order + 1,
                    dtype=torch.float32,
                )
                .expand(in_features, -1)
                .contiguous()
            )
            # Register buffer to make it part of the model's state and move to the correct device
            self.register_buffer(f"grid_{i}", grid)

        # Initialize weights using Kaiming uniform for better training dynamics
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity="linear")
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity="linear")

    def forward(self, x):
        """
        Forward pass through the two-layer KAN. Handles multi-dimensional inputs
        by reshaping them for internal 2D processing.
        """
        original_shape = x.shape
        # Reshape input to be 2D (batch_size * other_dims, features) to make
        # the internal logic compatible with multi-dimensional time-series data.
        x = x.reshape(-1, original_shape[-1])

        # Process each layer sequentially
        for i, (base_weight, spline_weight, layer_norm, prelu, dropout) in enumerate(
            zip(
                self.base_weights,
                self.spline_weights,
                self.layer_norms,
                self.prelus,
                self.dropouts,
            )
        ):
            # Ensure input tensor is on the same device as the model parameters
            x = x.to(base_weight.device)

            # Retrieve the grid for the current layer from the buffer
            grid = self._buffers[f"grid_{i}"]

            # 1. Base linear transformation (residual connection)
            # This works correctly as F.linear applies to the last dimension.
            base_output = F.linear(self.base_activation(x), base_weight)

            # 2. Spline transformation (learnable activation function)
            x_uns = x.unsqueeze(-1)

            # Calculate B-spline basis functions using the Cox-de Boor recursion formula
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                left_intervals = grid[:, : -(k + 1)]
                right_intervals = grid[:, k:-1]

                # Handle potential division by zero if grid points are identical
                delta_right = right_intervals - left_intervals
                delta_left = grid[:, k + 1 :] - grid[:, 1:(-k)]

                delta_right = torch.where(
                    delta_right == 0, torch.ones_like(delta_right), delta_right
                )
                delta_left = torch.where(
                    delta_left == 0, torch.ones_like(delta_left), delta_left
                )

                term1 = (x_uns - left_intervals) / delta_right * bases[:, :, :-1]
                term2 = (grid[:, k + 1 :] - x_uns) / delta_left * bases[:, :, 1:]
                bases = term1 + term2

            bases = bases.contiguous()

            # Apply spline weights. The views are now correct because the input `x` was reshaped.
            spline_output = F.linear(
                bases.view(x.size(0), -1), spline_weight.view(spline_weight.size(0), -1)
            )

            # 3. Combine, normalize, activate, and apply dropout
            x = base_output + spline_output
            x = layer_norm(x)
            x = prelu(x)
            x = dropout(x)

        # Reshape output back to the original input shape, preserving all dimensions
        # except for the last one, which is the new feature dimension.
        output_features = self.layers_hidden[-1]
        x = x.view(*original_shape[:-1], output_features)

        return x


class OldKAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        feature_dim,
        pred_len,
        bias=True,
        drop=0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode="gelu")
        self.fc2 = nn.Linear(pred_len, pred_len, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x).permute(0, 2, 1)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)
