import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def upsample_dataset(x, factor: int) -> torch.Tensor:
    aug_seq_len = factor * x.shape[0]
    # x: (T, C, H, W) -> need (T, H, W, C) for interpolate, then back
    x = x.permute(0, 2, 3, 1)  # (T, H, W, C)
    x = x.unsqueeze(0)  # (1, T, H, W, C)
    # Reshape for interpolate: (1, C, T, H, W)
    x = x.permute(0, 4, 1, 2, 3)
    x = F.interpolate(x, size=(aug_seq_len, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
    # Back to (T, C, H, W)
    x = x.squeeze(0).permute(1, 0, 2, 3)
    return x

class ShallowWaterPhysics(nn.Module):
    """Applies 2D shallow water equations with learnable physical parameters."""
    
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        
        # Learnable physical parameters (initialized with reasonable values)
        self.coriolis = nn.Parameter(torch.tensor(1e-4))  # Coriolis parameter (f)
        self.viscosity = nn.Parameter(torch.tensor(1e-3))  # Eddy viscosity (ν)
        self.friction = nn.Parameter(torch.tensor(1e-4))  # Bottom friction (r)
        self.dt = nn.Parameter(torch.tensor(0.1))  # Time step
        
        # Learnable mixing weights (how much physics vs learned correction)
        self.physics_weight = nn.Parameter(torch.tensor(0.5))
        
        # Neural network correction terms (residual learning)
        self.correction_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )
    
    def compute_derivatives(self, field):
        """Compute spatial derivatives using finite differences.
        
        Args:
            field: (C, H, W) tensor
        Returns:
            dx, dy: derivatives in x and y directions
        """
        # Central differences with periodic boundary conditions
        dx = (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / 2.0
        dy = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / 2.0
        return dx, dy
    
    def compute_laplacian(self, field):
        """Compute Laplacian (∇²) using finite differences.
        
        Args:
            field: (C, H, W) tensor
        Returns:
            laplacian: (C, H, W) tensor
        """
        # 5-point stencil for Laplacian
        center = field
        left = torch.roll(field, 1, dims=2)
        right = torch.roll(field, -1, dims=2)
        up = torch.roll(field, 1, dims=1)
        down = torch.roll(field, -1, dims=1)
        
        laplacian = left + right + up + down - 4 * center
        return laplacian
    
    def forward(self, velocity_field):
        """Apply shallow water equations to evolve velocity field.
        
        Args:
            velocity_field: (C, H, W) where C=2 for (u, v) components
        Returns:
            updated_field: (C, H, W) velocity field at next time step
        """
        u = velocity_field[0:1]  # (1, H, W)
        v = velocity_field[1:2]  # (1, H, W)
        
        # Compute spatial derivatives
        du_dx, du_dy = self.compute_derivatives(u)
        dv_dx, dv_dy = self.compute_derivatives(v)
        
        # Compute Laplacians for diffusion
        laplacian_u = self.compute_laplacian(u)
        laplacian_v = self.compute_laplacian(v)
        
        # Shallow water momentum equations (simplified)
        # ∂u/∂t = -u∂u/∂x - v∂u/∂y + f*v + ν∇²u - r*u
        # ∂v/∂t = -u∂v/∂x - v∂v/∂y - f*u + ν∇²v - r*v
        
        advection_u = -(u * du_dx + v * du_dy)
        advection_v = -(u * dv_dx + v * dv_dy)
        
        coriolis_u = self.coriolis * v
        coriolis_v = -self.coriolis * u
        
        diffusion_u = self.viscosity * laplacian_u
        diffusion_v = self.viscosity * laplacian_v
        
        friction_u = -self.friction * u
        friction_v = -self.friction * v
        
        # Total physics-based update
        du_dt_physics = advection_u + coriolis_u + diffusion_u + friction_u
        dv_dt_physics = advection_v + coriolis_v + diffusion_v + friction_v
        
        # Apply forward Euler time integration
        u_physics = u + self.dt * du_dt_physics
        v_physics = v + self.dt * dv_dt_physics
        
        # Stack for correction network
        velocity_physics = torch.cat([u_physics, v_physics], dim=0)  # (2, H, W)
        
        # Neural network correction (residual learning)
        velocity_physics_expanded = velocity_physics.unsqueeze(0)  # (1, 2, H, W)
        correction = self.correction_net(velocity_physics_expanded).squeeze(0)  # (2, H, W)
        
        # Blend physics and learned correction
        physics_contrib = torch.sigmoid(self.physics_weight) * velocity_physics
        learned_contrib = (1 - torch.sigmoid(self.physics_weight)) * (velocity_physics + correction)
        updated_field = physics_contrib + learned_contrib
        
        return updated_field

class SeaCurrentRNN(nn.Module):
    """A CNN encoder-decoder + Physics-informed RNN model using shallow water equations."""
    
    def __init__(self, hidden_size: int, input_height: int = 600, input_width: int = 600, output_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cnn_out_features = 256
        
        # Calculate encoded spatial dimensions
        # After 2 pooling layers with stride 2: H/4, W/4
        self.encoded_h = input_height // 4
        self.encoded_w = input_width // 4
        self.encoded_channels = 8
        self.encoded_flat_size = self.encoded_channels * self.encoded_h * self.encoded_w
        
        # CNN Encoder layers (downsample)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Physics module operating on encoded spatial features
        # We'll use a variant that works on the encoded channels
        self.physics_encoder = ShallowWaterPhysics(height=self.encoded_h, width=self.encoded_w)
        
        # Projection layers to/from physics space
        self.to_physics = nn.Conv2d(self.encoded_channels, 2, kernel_size=1)  # 8 -> 2 (u, v)
        self.from_physics = nn.Conv2d(2, self.encoded_channels, kernel_size=1)  # 2 -> 8
        
        # Recurrent hidden state processing
        self.W_in = nn.Linear(self.encoded_flat_size, hidden_size, bias=True)
        self.W_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_out = nn.Linear(hidden_size, self.encoded_flat_size, bias=True)
        
        # CNN Decoder layers (upsample)
        self.deconv1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
    
    def cnn_encode(self, x):
        # x shape: (T, H, W, C) -> need (T, C, H, W) for PyTorch
        x = x.permute(0, 3, 1, 2)  # (T, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # (T, 4, H/2, W/2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (T, 8, H/4, W/4)
        return x  # Return spatial features, not flattened
    
    def cnn_decode(self, x):
        # x: (T, 8, H/4, W/4) spatial features
        # Upsample with transposed convolutions
        x = F.relu(self.deconv1(x))  # (T, 4, H/2, W/2)
        x = F.relu(self.deconv2(x))  # (T, 2, H, W)
        x = self.conv_final(x)  # (T, 2, H, W)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, H, W, C)
        
        # Encode through CNN
        x_encoded = self.cnn_encode(x)  # (T, 8, H/4, W/4)
        
        # Physics-informed RNN processing
        batch_size = x_encoded.shape[0]
        h = torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype)
        spatial_states = []
        
        for t in range(batch_size):
            x_t = x_encoded[t]  # (8, H/4, W/4)
            
            # Project to physics space (u, v velocity components)
            velocity_field = self.to_physics(x_t.unsqueeze(0)).squeeze(0)  # (2, H/4, W/4)
            
            # Apply shallow water equations
            velocity_evolved = self.physics_encoder(velocity_field)  # (2, H/4, W/4)
            
            # Project back to feature space
            features_evolved = self.from_physics(velocity_evolved.unsqueeze(0)).squeeze(0)  # (8, H/4, W/4)
            
            # Maintain hidden state for memory (processes flattened features)
            features_flat = features_evolved.reshape(-1)  # (8*H/4*W/4,)
            input_logits = self.W_in(features_flat)
            memory_logits = self.W_hidden(h)
            h = torch.tanh(input_logits + memory_logits)
            correction = self.W_out(h)  # (8*H/4*W/4,)
            
            # Combine physics evolution with learned correction
            correction_spatial = correction.reshape(self.encoded_channels, self.encoded_h, self.encoded_w)
            output_state = features_evolved + 0.1 * correction_spatial  # Residual connection
            
            spatial_states.append(output_state)
        
        outputs = torch.stack(spatial_states)  # (T, 8, H/4, W/4)
        
        # Decode through CNN
        outputs = self.cnn_decode(outputs)  # (T, 2, H, W)
        
        # Convert back to (T, H, W, C) to match input format
        outputs = outputs.permute(0, 2, 3, 1)  # (T, H, W, 2)
        
        return outputs
