import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Positional Encoding Module
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Adds positional encoding to the input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on
        # position and dimension.
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [bs, seq_length, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ----------------------------
# Residual Block Module
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        """
        A simple residual block with two fully-connected layers.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x shape: [seq_length, bs, hidden_size]
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + residual)

# ----------------------------
# Multimodal Gated Residual Network (MultimodalGRN) with Pangu Support
# ----------------------------
class MultimodalGRN(nn.Module):
    def __init__(self, config):
        """
        Multimodal GRN with cross-attention fusion mechanism for Pangu data.
        
        Config structure expected for MultimodalGRN:
        {
            "model": {
                "params": {
                    "hidden_size": int,
                    "dropout": float,
                    "fusion_weight": float,
                    "num_heads": int,
                    "pangu_spatial_H": int,
                    "pangu_spatial_W": int,
                    "pangu1_time_dim": int,
                    "pangu1_height_dim": int,
                    "pangu1_features_dim": int,
                    "pangu2_time_dim": int,
                    "pangu2_features_dim": int,
                    "pangu_output_ratio": float
                }
            }
        }
        """
        super(MultimodalGRN, self).__init__()
        
        model_params = config['model']['params']
        self.hidden_size = model_params['hidden_size']
        dropout = model_params['dropout']
        self.fusion_weight = None
        num_heads = model_params['num_heads']
        
        # Pangu processing parameters
        pangu_spatial_dim_H = model_params.get('pangu_spatial_H', 11)
        pangu_spatial_dim_W = model_params.get('pangu_spatial_W', 11)
        pangu_spatial_flat_dim = pangu_spatial_dim_H * pangu_spatial_dim_W
        
        # Pangu dimensions
        self.time_p1 = model_params.get('pangu1_time_dim', 2)
        self.height_p1 = model_params.get('pangu1_height_dim', 5)
        self.features_p1 = model_params.get('pangu1_features_dim', 13)
        
        self.time_p2 = model_params.get('pangu2_time_dim', 2)
        self.features_p2 = model_params.get('pangu2_features_dim', 4)
        
        # Calculate pangu feature sizes
        pangu1_features_after_spatial = self.height_p1 * self.features_p1
        pangu2_features_after_spatial = self.features_p2
        pangu_combined_fc_input_dim = pangu1_features_after_spatial + pangu2_features_after_spatial
        
        pangu_output_ratio = model_params['pangu_output_ratio']
        self.pangu_feature_size = math.floor(self.hidden_size * pangu_output_ratio)
        if self.pangu_feature_size == 0 and self.hidden_size > 0 and pangu_output_ratio > 0:
            self.pangu_feature_size = 1
        
        # Spatial processing for pangu and fengwu
        self.pangu_spatial_processor = nn.Linear(pangu_spatial_flat_dim, 1)
        self.pangu_surface_spatial_processor = nn.Linear(pangu_spatial_flat_dim, 1)

        # Pangu feature processing
        self.fc_pangu = nn.Sequential(
            nn.Linear(pangu_combined_fc_input_dim, self.pangu_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pangu_temporal_upsampler = nn.ConvTranspose1d(
            self.pangu_feature_size, self.pangu_feature_size, kernel_size=6, stride=6
        )

        # Fengwu feature processing (separate FC layer for Fengwu data)
        # Fengwu has 69 features after spatial processing
        fengwu_features = model_params.get('fengwu_features', 69)
        self.fc_fengwu = nn.Sequential(
            nn.Linear(fengwu_features, self.pangu_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Projection layers for both modalities to same hidden space
        self.x_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.y_proj = nn.Linear(self.pangu_feature_size, self.hidden_size)
        
        # Cross-attention mechanism
        self.x_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.y_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Feature fusion layer
        self.fusion_fc = nn.Linear(self.hidden_size * 2, 1)
    
    def process_pangu_tensor(self, pangu_tensor, expected_dims_before_spatial):
        """
        Helper function to process a single pangu tensor.
        Expected input shape: [bs, dim1, dim2, ..., H, W]
        expected_dims_before_spatial: tuple of (dim1, dim2, ...)
        """
        bs = pangu_tensor.shape[0]
        spatial_H = pangu_tensor.shape[-2]
        spatial_W = pangu_tensor.shape[-1]

        # Reshape for spatial processing: [bs * other_dims_product, H * W]
        other_dims_prod = 1
        for i in range(1, len(pangu_tensor.shape) - 2):
            other_dims_prod *= pangu_tensor.shape[i]
        
        pangu_flat_spatial = pangu_tensor.reshape(bs * other_dims_prod, spatial_H * spatial_W)
        
        # Process spatially: [bs * other_dims_product, 1]
        if len(expected_dims_before_spatial) == 3:
            pangu_reduced_spatial = self.pangu_spatial_processor(pangu_flat_spatial)
        else:
            pangu_reduced_spatial = self.pangu_surface_spatial_processor(pangu_flat_spatial)
        
        # Reshape back and remove the last dim (which is now 1):
        new_shape = (bs,) + expected_dims_before_spatial
        processed_pangu = pangu_reduced_spatial.reshape(new_shape)
        
        # Flatten all non-batch dimensions: [bs, product_of_other_dims]
        return processed_pangu.reshape(bs, -1)
    
    def process_pangu_data(self, pangu1, pangu2):
        """
        Process pangu1 and pangu2 data and combine them.
        pangu1: [bs, time_p1, height_p1, features_p1, H, W]
        pangu2: [bs, time_p2, features_p2, H, W]
        """
        # Process pangu1 and pangu2
        pangu1_processed = self.process_pangu_tensor(pangu1,
                                                     (self.time_p1, self.height_p1, self.features_p1))
        pangu2_processed = self.process_pangu_tensor(pangu2,
                                                     (self.time_p2, self.features_p2))

        # Concatenate processed pangu features
        pangu1_processed = pangu1_processed.reshape(pangu1.shape[0], self.time_p1, -1)
        pangu2_processed = pangu2_processed.reshape(pangu2.shape[0], self.time_p2, -1)
        pangu_combined_features = torch.cat((pangu1_processed, pangu2_processed), dim=-1)

        # Pass through the Pangu FC layer
        pangu_features = self.fc_pangu(pangu_combined_features)
        pangu_aligned = self.pangu_temporal_upsampler(pangu_features.transpose(1, 2))
        pangu_final_vec = pangu_aligned.mean(dim=2)

        return pangu_final_vec

    def process_fengwu_data(self, fengwu):
        """
        Process fengwu data.
        fengwu: [bs, features, H, W] - typically [bs, 69, 11, 11]
        """
        bs = fengwu.shape[0]
        features = fengwu.shape[1]
        spatial_H = fengwu.shape[2]
        spatial_W = fengwu.shape[3]

        # Reshape for spatial processing: [bs * features, H * W]
        fengwu_flat_spatial = fengwu.reshape(bs * features, spatial_H * spatial_W)

        # Process spatially: [bs * features, 1]
        fengwu_reduced_spatial = self.pangu_spatial_processor(fengwu_flat_spatial)

        # Reshape back: [bs, features]
        fengwu_processed = fengwu_reduced_spatial.reshape(bs, features)

        # Pass through the Fengwu-specific FC layer
        fengwu_final_vec = self.fc_fengwu(fengwu_processed)

        return fengwu_final_vec
        
    def forward(self, x, pangu1, pangu2=None):
        """
        Args:
            x: [bs, hidden_size] - primary modality (from transformer)
            pangu1: [bs, time_p1, height_p1, features_p1, H, W] - first pangu modality
                    OR [bs, features, H, W] - fengwu data
            pangu2: [bs, time_p2, features_p2, H, W] - second pangu modality (optional, None for Fengwu)
        Returns:
            fused_x: [bs, hidden_size] - enhanced x features
            fused_y: [bs, pangu_feature_size] - enhanced weather features
        """
        bs = x.shape[0]

        # Detect data type and process accordingly
        if pangu2 is None:
            # Fengwu mode: only one weather input
            y = self.process_fengwu_data(pangu1)
        else:
            # Pangu mode: two weather inputs
            y = self.process_pangu_data(pangu1, pangu2)

        
        # Project both modalities to same hidden space
        x_hidden = F.relu(self.x_proj(x))  # [bs, hidden_size]
        y_hidden = F.relu(self.y_proj(y))  # [bs, hidden_size]
        
        # Add sequence dimension for attention
        x_seq = x_hidden.unsqueeze(1)  # [bs, 1, hidden_size]
        y_seq = y_hidden.unsqueeze(1)  # [bs, 1, hidden_size]
        
        # Cross-attention: let x attend to y and vice versa
        x_attended, _ = self.x_attention(x_seq, y_seq, y_seq)  # x queries, y keys/values
        y_attended, _ = self.y_attention(y_seq, x_seq, x_seq)  # y queries, x keys/values
        
        # Remove sequence dimension
        x_attended = x_attended.squeeze(1)  # [bs, hidden_size]
        y_attended = y_attended.squeeze(1)  # [bs, hidden_size]
        
        gate_input = torch.cat((x_attended, y_attended), dim=-1)
        alpha = torch.sigmoid(self.fusion_fc(gate_input))
        if self.fusion_weight is not None:
            alpha = torch.full_like(alpha, self.fusion_weight)
        fused_hidden = (1 - alpha) * x_attended + alpha * y_attended
        
        return fused_hidden, y
    
    def set_fusion_weight(self, weight):
        """Dynamically adjust fusion weight"""
        self.fusion_weight = weight





# ----------------------------
# GRCMI-WPF Model Definition with Pangu Multimodal Support
# ----------------------------
class GRCMI_WPF(nn.Module):
    def __init__(self, config):
        """
        GRCMI-WPF Model with Multimodal GRN for Pangu data:
          - Input projection maps [feature] to [hidden_size]
          - Positional encoding is added to the input embeddings
          - Transformer encoder layers capture temporal dependencies
          - A residual block further processes the sequence
          - A MultimodalGRN aggregates the last timestamp's information with pangu modalities
          - A final linear decoder produces [bs, prediction_length] output.
        
        Config structure expected:
        {
            "turbine_indices": [...],
            "feature_indices": [...],
            "prediction_horizon": int,
            "model": {
                "params": {
                    "num_layers": int,
                    "hidden_size": int,
                    "num_heads": int,
                    "dropout": float,
                    "fusion_weight": float,
                    "max_len": int (optional),
                    "pangu_spatial_H": int,
                    "pangu_spatial_W": int,
                    "pangu1_time_dim": int,
                    "pangu1_height_dim": int,
                    "pangu1_features_dim": int,
                    "pangu2_time_dim": int,
                    "pangu2_features_dim": int,
                    "pangu_output_ratio": float
                }
            }
        }
        """
        super(GRCMI_WPF, self).__init__()
        
        # Extract all parameters from config
        self.input_feature_size = len(config["turbine_indices"]) * len(config["feature_indices"])
        self.prediction_length = config["prediction_horizon"]
        
        model_params = config['model']['params']
        self.hidden_size = model_params['hidden_size']
        num_encoder_layers = model_params['num_layers']
        num_heads = model_params['num_heads']
        dropout = model_params['dropout']
        max_len = model_params.get('max_len', 5000)
        
        # Project the input from feature dimensions to hidden_size
        self.input_proj = nn.Linear(self.input_feature_size, self.hidden_size)
        
        # Add positional encoding so that model knows position information.
        self.positional_encoding = PositionalEncoding(self.hidden_size, dropout, max_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Residual block on the sequence output from the encoder
        self.residual_block = ResidualBlock(self.hidden_size)
        
        # Multimodal Gated Residual Network to further process aggregated representation.
        self.multimodal_grn = MultimodalGRN(config)
        
        # Final linear layer to map the processed representation to the prediction length.
        self.decoder = nn.Linear(self.hidden_size, self.prediction_length)
    
    def forward(self, x, *args):
        """
        Forward pass for GRCMI-WPF with weather data support.
        Input:
            x: Primary modality tensor with shape [bs, seq_length, turbines, features]
            args[0]: weather data - either:
                     - pangu1 tensor with shape [bs, time_p1, height_p1, features_p1, H, W] (Pangu mode)
                     - fengwu tensor with shape [bs, features, H, W] (Fengwu mode)
            args[1]: pangu2 tensor with shape [bs, time_p2, features_p2, H, W] (Pangu mode only, optional)
        Output:
            A tensor with shape [bs, prediction_length]
        """
        bs, seq, wt, p = x.shape
        x = x.reshape(bs, seq, -1)
        bs, seq_length, _ = x.size()

        # Extract weather data - support both Pangu (2 args) and Fengwu (1 arg)
        if len(args) == 1:
            # Fengwu mode: only one weather input
            weather1 = args[0]
            weather2 = None
        else:
            # Pangu mode: two weather inputs
            weather1, weather2 = args[0], args[1]
        
        # Step 1: Project input feature dimension to hidden size
        x = self.input_proj(x)  # x now has shape: [bs, seq_length, hidden_size]
        
        # Step 2: Positional Encoding to inject sequence order information
        x = self.positional_encoding(x)
        
        # Step 3: Transformer expects input shape [seq_length, bs, hidden_size]
        x = x.permute(1, 0, 2)
        
        # Step 4: Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # output shape: [seq_length, bs, hidden_size]
        
        # Step 5: Apply the Residual Block for additional non-linear processing
        x = self.residual_block(x)
        
        # Step 6: Bring the tensor back to [bs, seq_length, hidden_size]
        x = x.permute(1, 0, 2)
        
        # Step 7: Aggregate the temporal information.
        # Here, we simply take the representation of the last time step.
        x_last = x[:, -1, :]  # shape: [bs, hidden_size]
        
        # Step 8: Process with Multimodal GRN using weather data (Pangu or Fengwu)
        x_grn, y_grn = self.multimodal_grn(x_last, weather1, weather2)
        
        # Step 9: Final decoder projects the hidden representation to the output dimension.
        output = self.decoder(x_grn)  # shape: [bs, prediction_length]
        
        return output
    
    def set_fusion_weight(self, weight):
        """Set the fusion weight for multimodal GRN"""
        self.multimodal_grn.set_fusion_weight(weight)


# Multimodal Gated Residual Network (MultimodalGRN) with Pangu Support
# ----------------------------
# class MultimodalGRN_Ada(nn.Module):
#     def __init__(self, config):
#         """
#         Multimodal GRN with cross-attention fusion mechanism for Pangu data.
#
#         Config structure expected for MultimodalGRN:
#         {
#             "model": {
#                 "params": {
#                     "hidden_size": int,
#                     "dropout": float,
#                     "fusion_weight": float,
#                     "num_heads": int,
#                     "pangu_spatial_H": int,
#                     "pangu_spatial_W": int,
#                     "pangu1_time_dim": int,
#                     "pangu1_height_dim": int,
#                     "pangu1_features_dim": int,
#                     "pangu2_time_dim": int,
#                     "pangu2_features_dim": int,
#                     "pangu_output_ratio": float
#                 }
#             }
#         }
#         """
#         super(MultimodalGRN_Ada, self).__init__()
#
#         model_params = config['model']['params']
#         self.hidden_size = model_params['hidden_size']
#         dropout = model_params['dropout']
#         # 将fusion_weight改为可学习参数，初始值使用配置中的值
#         initial_fusion_weight = model_params['fusion_weight']
#         self.fusion_weight = nn.Parameter(torch.tensor(initial_fusion_weight))
#         self.fusion_weight_fc=nn.Linear(self.hidden_size *2,1)
#         num_heads = model_params['num_heads']
#
#         # Pangu processing parameters
#         pangu_spatial_dim_H = model_params.get('pangu_spatial_H', 11)
#         pangu_spatial_dim_W = model_params.get('pangu_spatial_W', 11)
#         pangu_spatial_flat_dim = pangu_spatial_dim_H * pangu_spatial_dim_W
#
#         # Pangu dimensions
#         self.time_p1 = model_params.get('pangu1_time_dim', 2)
#         self.height_p1 = model_params.get('pangu1_height_dim', 5)
#         self.features_p1 = model_params.get('pangu1_features_dim', 13)
#
#         self.time_p2 = model_params.get('pangu2_time_dim', 2)
#         self.features_p2 = model_params.get('pangu2_features_dim', 4)
#
#         # Calculate pangu feature sizes
#         pangu1_features_after_spatial = self.time_p1 * self.height_p1 * self.features_p1
#         pangu2_features_after_spatial = self.time_p2 * self.features_p2
#         pangu_combined_fc_input_dim = pangu1_features_after_spatial + pangu2_features_after_spatial
#
#         pangu_output_ratio = model_params['pangu_output_ratio']
#         self.pangu_feature_size = math.floor(self.hidden_size * pangu_output_ratio)
#         if self.pangu_feature_size == 0 and self.hidden_size > 0 and pangu_output_ratio > 0:
#             self.pangu_feature_size = 1
#
#         # Spatial processing for pangu
#         self.pangu_spatial_processor = nn.Linear(pangu_spatial_flat_dim, 1)
#
#         # Pangu feature processing
#         self.fc_pangu = nn.Sequential(
#             nn.Linear(pangu_combined_fc_input_dim, self.pangu_feature_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         # Projection layers for both modalities to same hidden space
#         self.x_proj = nn.Linear(self.hidden_size, self.hidden_size)
#         self.y_proj = nn.Linear(self.pangu_feature_size, self.hidden_size)
#
#         # Cross-attention mechanism
#         self.x_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout,
#                                                  batch_first=True)
#         self.y_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout,
#                                                  batch_first=True)
#
#         # Feature fusion layer
#         self.fusion_fc = nn.Linear(self.hidden_size * 2, self.hidden_size)
#
#         # GRN processing layers
#         self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc2_x = nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc2_y = nn.Linear(self.hidden_size, self.pangu_feature_size)
#
#         self.dropout = nn.Dropout(dropout)
#         self.sigmoid = nn.Sigmoid()
#         self.layer_norm = nn.LayerNorm(self.hidden_size)
#
#     def process_pangu_tensor(self, pangu_tensor, expected_dims_before_spatial):
#         """
#         Helper function to process a single pangu tensor.
#         Expected input shape: [bs, dim1, dim2, ..., H, W]
#         expected_dims_before_spatial: tuple of (dim1, dim2, ...)
#         """
#         bs = pangu_tensor.shape[0]
#         spatial_H = pangu_tensor.shape[-2]
#         spatial_W = pangu_tensor.shape[-1]
#
#         # Reshape for spatial processing: [bs * other_dims_product, H * W]
#         other_dims_prod = 1
#         for i in range(1, len(pangu_tensor.shape) - 2):
#             other_dims_prod *= pangu_tensor.shape[i]
#
#         pangu_flat_spatial = pangu_tensor.reshape(bs * other_dims_prod, spatial_H * spatial_W)
#
#         # Process spatially: [bs * other_dims_product, 1]
#         pangu_reduced_spatial = self.pangu_spatial_processor(pangu_flat_spatial)
#
#         # Reshape back and remove the last dim (which is now 1):
#         new_shape = (bs,) + expected_dims_before_spatial
#         processed_pangu = pangu_reduced_spatial.reshape(new_shape)
#
#         # Flatten all non-batch dimensions: [bs, product_of_other_dims]
#         return processed_pangu.reshape(bs, -1)
#
#     def process_pangu_data(self, pangu1, pangu2):
#         """
#         Process pangu1 and pangu2 data and combine them.
#         pangu1: [bs, time_p1, height_p1, features_p1, H, W]
#         pangu2: [bs, time_p2, features_p2, H, W]
#         """
#         # Process pangu1 and pangu2
#         pangu1_processed = self.process_pangu_tensor(pangu1,
#                                                      (self.time_p1, self.height_p1, self.features_p1))
#         pangu2_processed = self.process_pangu_tensor(pangu2,
#                                                      (self.time_p2, self.features_p2))
#
#         # Concatenate processed pangu features
#         pangu_combined_features = torch.cat((pangu1_processed, pangu2_processed), dim=1)
#
#         # Pass through the Pangu FC layer
#         pangu_final_vec = self.fc_pangu(pangu_combined_features)
#
#         return pangu_final_vec
#
#     def forward(self, x, pangu1, pangu2):
#         """
#         Args:
#             x: [bs, hidden_size] - primary modality (from transformer)
#             pangu1: [bs, time_p1, height_p1, features_p1, H, W] - first pangu modality
#             pangu2: [bs, time_p2, features_p2, H, W] - second pangu modality
#         Returns:
#             fused_x: [bs, hidden_size] - enhanced x features
#             fused_y: [bs, pangu_feature_size] - enhanced pangu features
#         """
#         bs = x.shape[0]
#         x_residual = x
#
#         # Process pangu data
#         y = self.process_pangu_data(pangu1, pangu2)
#         y_residual = y
#
#         # Project both modalities to same hidden space
#         x_hidden = F.relu(self.x_proj(x))  # [bs, hidden_size]
#         y_hidden = F.relu(self.y_proj(y))  # [bs, hidden_size]
#
#         # Add sequence dimension for attention
#         x_seq = x_hidden.unsqueeze(1)  # [bs, 1, hidden_size]
#         y_seq = y_hidden.unsqueeze(1)  # [bs, 1, hidden_size]
#
#         # Cross-attention: let x attend to y and vice versa
#         x_attended, _ = self.x_attention(x_seq, y_seq, y_seq)  # x queries, y keys/values
#         y_attended, _ = self.y_attention(y_seq, x_seq, x_seq)  # y queries, x keys/values
#
#         # Remove sequence dimension
#         x_attended = x_attended.squeeze(1)  # [bs, hidden_size]
#         y_attended = y_attended.squeeze(1)  # [bs, hidden_size]
#
#
#         self.fusion_weight1=self.fusion_weight_fc(torch.cat((x_attended,y_attended),dim=1))
#         # Fusion with learnable weight control (使用sigmoid确保权重在0-1之间)
#         fusion_weight_norm = torch.sigmoid(self.fusion_weight1)
#
#         # Multi-level fusion
#         x_enhanced = fusion_weight_norm * x_hidden + (1 - fusion_weight_norm) * x_attended
#         y_enhanced = (1 - fusion_weight_norm) * y_hidden + fusion_weight_norm * y_attended
#
#         # Concatenate and fuse features
#         fused_features = torch.cat([x_enhanced, y_enhanced], dim=-1)  # [bs, hidden_size*2]
#         fused_hidden = F.relu(self.fusion_fc(fused_features))  # [bs, hidden_size]
#         fused_hidden = self.layer_norm(fused_hidden)
#
#         # GRN processing
#         processed = F.relu(self.fc1(fused_hidden))
#         processed = self.dropout(processed)
#
#         # Project back to original dimensions
#         x_output = self.fc2_x(processed)  # [bs, hidden_size]
#         y_output = self.fc2_y(processed)  # [bs, pangu_feature_size]
#
#         # Gating mechanism
#         x_gate = self.sigmoid(x_output)
#         y_gate = self.sigmoid(y_output)
#
#         # Apply gates with residual connections
#         fused_x = x_gate * x_residual + x_residual
#         fused_y = y_gate * y_residual + y_residual
#
#         return fused_x, fused_y
#
#     def set_fusion_weight(self, weight):
#         """Dynamically adjust fusion weight"""
#         with torch.no_grad():
#             self.fusion_weight.data = torch.tensor(weight)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultimodalGRN_Ada(nn.Module):
    def __init__(self, config):
        """
        Multimodal GRN with a clearer, adaptive cross-attention fusion mechanism.

        This refactored version clarifies the fusion logic to act as a true "valve"
        controlling the blend of the two modalities.

        Config structure expected:
        {
            "model": {
                "params": {
                    "hidden_size": int,
                    "dropout": float,
                    "num_heads": int,
                    "pangu_spatial_H": int,
                    "pangu_spatial_W": int,
                    "pangu1_time_dim": int,
                    "pangu1_height_dim": int,
                    "pangu1_features_dim": int,
                    "pangu2_time_dim": int,
                    "pangu2_features_dim": int,
                    "pangu_output_ratio": float
                }
            }
        }
        """
        super(MultimodalGRN_Ada, self).__init__()

        model_params = config['model']['params']
        self.hidden_size = model_params['hidden_size']
        dropout = model_params['dropout']
        num_heads = model_params['num_heads']

        # Pangu processing parameters
        pangu_spatial_dim_H = model_params.get('pangu_spatial_H', 11)
        pangu_spatial_dim_W = model_params.get('pangu_spatial_W', 11)
        pangu_spatial_flat_dim = pangu_spatial_dim_H * pangu_spatial_dim_W

        # Pangu dimensions
        self.time_p1 = model_params.get('pangu1_time_dim', 2)
        self.height_p1 = model_params.get('pangu1_height_dim', 5)
        self.features_p1 = model_params.get('pangu1_features_dim', 13)

        self.time_p2 = model_params.get('pangu2_time_dim', 2)
        self.features_p2 = model_params.get('pangu2_features_dim', 4)

        # Calculate pangu feature sizes
        pangu1_features_after_spatial = self.time_p1 * self.height_p1 * self.features_p1
        pangu2_features_after_spatial = self.time_p2 * self.features_p2
        pangu_combined_fc_input_dim = pangu1_features_after_spatial + pangu2_features_after_spatial

        pangu_output_ratio = model_params['pangu_output_ratio']
        self.pangu_feature_size = math.floor(self.hidden_size * pangu_output_ratio)
        if self.pangu_feature_size == 0 and self.hidden_size > 0 and pangu_output_ratio > 0:
            self.pangu_feature_size = 1

        # Spatial processing for pangu
        self.pangu_spatial_processor = nn.Linear(pangu_spatial_flat_dim, 1)

        # Pangu feature processing
        self.fc_pangu = nn.Sequential(
            nn.Linear(pangu_combined_fc_input_dim, self.pangu_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Projection layers for both modalities to the same hidden space
        self.x_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.y_proj = nn.Linear(self.pangu_feature_size, self.hidden_size)

        # Cross-attention mechanism
        self.x_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout,
                                                 batch_first=True)
        self.y_attention = nn.MultiheadAttention(self.hidden_size, num_heads=num_heads, dropout=dropout,
                                                 batch_first=True)

        # --- REFACTORED FUSION LOGIC ---
        # This layer creates the adaptive gate (the "valve") from the concatenated
        # attended features. Its output is a single value per item in the batch.
        self.fusion_gate_fc = nn.Linear(self.hidden_size * 2, 1)
        # --- END REFACTORED FUSION LOGIC ---

        # GRN processing layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_x = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_y = nn.Linear(self.hidden_size, self.pangu_feature_size)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def process_pangu_tensor(self, pangu_tensor, expected_dims_before_spatial):
        # This helper function is well-designed and remains unchanged.
        bs = pangu_tensor.shape[0]
        spatial_H = pangu_tensor.shape[-2]
        spatial_W = pangu_tensor.shape[-1]
        other_dims_prod = 1
        for i in range(1, len(pangu_tensor.shape) - 2):
            other_dims_prod *= pangu_tensor.shape[i]
        pangu_flat_spatial = pangu_tensor.reshape(bs * other_dims_prod, spatial_H * spatial_W)
        pangu_reduced_spatial = self.pangu_spatial_processor(pangu_flat_spatial)
        new_shape = (bs,) + expected_dims_before_spatial
        processed_pangu = pangu_reduced_spatial.reshape(new_shape)
        return processed_pangu.reshape(bs, -1)

    def process_pangu_data(self, pangu1, pangu2):
        # This function is also well-designed and remains unchanged.
        pangu1_processed = self.process_pangu_tensor(pangu1,
                                                     (self.time_p1, self.height_p1, self.features_p1))
        pangu2_processed = self.process_pangu_tensor(pangu2,
                                                     (self.time_p2, self.features_p2))
        pangu_combined_features = torch.cat((pangu1_processed, pangu2_processed), dim=1)
        pangu_final_vec = self.fc_pangu(pangu_combined_features)
        return pangu_final_vec

    def forward(self, x, pangu1, pangu2):
        """
        Args:
            x: [bs, hidden_size] - primary modality (from transformer)
            pangu1: [bs, time_p1, height_p1, features_p1, H, W] - first pangu modality
            pangu2: [bs, time_p2, features_p2, H, W] - second pangu modality
        Returns:
            fused_x: [bs, hidden_size] - enhanced x features
            fused_y: [bs, pangu_feature_size] - enhanced pangu features
        """
        x_residual = x

        # 1. Process pangu data into a single vector 'y'
        y = self.process_pangu_data(pangu1, pangu2)
        y_residual = y

        # 2. Project both modalities to the same hidden dimension for attention
        x_hidden = F.relu(self.x_proj(x))
        y_hidden = F.relu(self.y_proj(y))

        # Add a sequence dimension (of length 1) for MultiheadAttention
        x_seq = x_hidden.unsqueeze(1)
        y_seq = y_hidden.unsqueeze(1)

        # 3. Perform cross-attention. Each modality attends to the other.
        # x_attended now contains information from y
        x_attended, _ = self.x_attention(query=x_seq, key=y_seq, value=y_seq)
        # y_attended now contains information from x
        y_attended, _ = self.y_attention(query=y_seq, key=x_seq, value=x_seq)

        # Remove the sequence dimension
        x_attended = x_attended.squeeze(1)
        y_attended = y_attended.squeeze(1)

        # --- REFACTORED ADAPTIVE FUSION ---
        # 4. Create the adaptive gate (the "valve")
        # Concatenate the two informed representations
        gate_input = torch.cat((x_attended, y_attended), dim=1)
        # Pass through a linear layer and sigmoid to get a gating value between 0 and 1
        # This value, alpha, determines the fusion ratio for each sample in the batch.
        alpha = torch.sigmoid(self.fusion_gate_fc(gate_input)) # Shape: [bs, 1]

        # 5. Fuse the representations using the adaptive gate 'alpha'
        # This is a clear, weighted average where 'alpha' controls the balance.
        # If alpha is close to 1, the fused representation relies more on x_attended.
        # If alpha is close to 0, it relies more on y_attended.
        fused_hidden = alpha * x_attended + (1 - alpha) * y_attended
        # print(alpha)
        # --- END REFACTORED ADAPTIVE FUSION ---

        # Normalize the fused representation
        fused_hidden = self.layer_norm(fused_hidden)

        # 6. Pass the fused representation through the rest of the GRN
        processed = F.relu(self.fc1(fused_hidden))
        processed = self.dropout(processed)

        # Project back to original modality dimensions
        x_output = self.fc2_x(processed)
        y_output = self.fc2_y(processed)

        # 7. Apply the final gating mechanism with residual connections
        # The gate's strength is determined by the fused information ('processed')
        # and it's applied to the original, unprocessed residual.
        x_gate = self.sigmoid(x_output)
        y_gate = self.sigmoid(y_output)

        fused_x = x_output + x_residual
        fused_y = y_gate * y_residual + y_residual

        return fused_x, fused_y,alpha




# ----------------------------
class GRCMI_WPF_Adaptive(nn.Module):
    def __init__(self, config):
        """
        GRCMI-WPF Model with Multimodal GRN for Pangu data:
          - Input projection maps [feature] to [hidden_size]
          - Positional encoding is added to the input embeddings
          - Transformer encoder layers capture temporal dependencies
          - A residual block further processes the sequence
          - A MultimodalGRN aggregates the last timestamp's information with pangu modalities
          - A final linear decoder produces [bs, prediction_length] output.

        Config structure expected:
        {
            "turbine_indices": [...],
            "feature_indices": [...],
            "prediction_horizon": int,
            "model": {
                "params": {
                    "num_layers": int,
                    "hidden_size": int,
                    "num_heads": int,
                    "dropout": float,
                    "fusion_weight": float,
                    "max_len": int (optional),
                    "pangu_spatial_H": int,
                    "pangu_spatial_W": int,
                    "pangu1_time_dim": int,
                    "pangu1_height_dim": int,
                    "pangu1_features_dim": int,
                    "pangu2_time_dim": int,
                    "pangu2_features_dim": int,
                    "pangu_output_ratio": float
                }
            }
        }
        """
        super(GRCMI_WPF_Adaptive, self).__init__()

        # Extract all parameters from config
        self.input_feature_size = len(config["turbine_indices"]) * len(config["feature_indices"])
        self.prediction_length = config["prediction_horizon"]

        model_params = config['model']['params']
        self.hidden_size = model_params['hidden_size']
        num_encoder_layers = model_params['num_layers']
        num_heads = model_params['num_heads']
        dropout = model_params['dropout']
        max_len = model_params.get('max_len', 5000)

        # Project the input from feature dimensions to hidden_size
        self.input_proj = nn.Linear(self.input_feature_size, self.hidden_size)

        # Add positional encoding so that model knows position information.
        self.positional_encoding = PositionalEncoding(self.hidden_size, dropout, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Residual block on the sequence output from the encoder
        self.residual_block = ResidualBlock(self.hidden_size)

        # Multimodal Gated Residual Network to further process aggregated representation.
        self.multimodal_grn = MultimodalGRN_Ada(config)

        # Final linear layer to map the processed representation to the prediction length.
        self.decoder = nn.Linear(self.hidden_size, self.prediction_length)

    def forward(self, x, *args):
        """
        Forward pass for GRCMI-WPF with pangu multimodal support.
        Input:
            x: Primary modality tensor with shape [bs, seq_length, turbines, features]
            args[0]: pangu1 tensor with shape [bs, time_p1, height_p1, features_p1, H, W]
            args[1]: pangu2 tensor with shape [bs, time_p2, features_p2, H, W]
        Output:
            A tensor with shape [bs, prediction_length]
        """
        bs, seq, wt, p = x.shape
        x = x.reshape(bs, seq, -1)
        bs, seq_length, _ = x.size()

        # Extract pangu data
        pangu1, pangu2 = args[0], args[1]

        # Step 1: Project input feature dimension to hidden size
        x = self.input_proj(x)  # x now has shape: [bs, seq_length, hidden_size]

        # Step 2: Positional Encoding to inject sequence order information
        x = self.positional_encoding(x)

        # Step 3: Transformer expects input shape [seq_length, bs, hidden_size]
        x = x.permute(1, 0, 2)

        # Step 4: Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # output shape: [seq_length, bs, hidden_size]

        # Step 5: Apply the Residual Block for additional non-linear processing
        x = self.residual_block(x)

        # Step 6: Bring the tensor back to [bs, seq_length, hidden_size]
        x = x.permute(1, 0, 2)

        # Step 7: Aggregate the temporal information.
        # Here, we simply take the representation of the last time step.
        x_last = x[:, -1, :]  # shape: [bs, hidden_size]

        # Step 8: Process with Multimodal GRN using pangu data
        x_grn, y_grn, alpha = self.multimodal_grn(x_last, pangu1, pangu2)

        # Step 9: Final decoder projects the hidden representation to the output dimension.
        output = self.decoder(x_grn)  # shape: [bs, prediction_length]

        return output,alpha

    def set_fusion_weight(self, weight):
        """Set the fusion weight for multimodal GRN"""
        self.multimodal_grn.set_fusion_weight(weight)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Example config structure with all hyperparameters for pangu processing
    config = {
        "turbine_indices": list(range(1)),              # 1 turbine
        "feature_indices": list(range(13)),             # 13 features per turbine
        "prediction_horizon": 12,                       # Predict 12 time steps
        "model": {
            "params": {
                "num_layers": 2,                        # Number of transformer encoder layers
                "hidden_size": 64,                      # Hidden dimension size
                "num_heads": 8,                         # Number of attention heads
                "dropout": 0.1,                         # Dropout rate
                "fusion_weight": 0.7,                   # Fusion weight (0.7 favors primary modality)
                "max_len": 1000,                        # Maximum sequence length for positional encoding
                "pangu_spatial_H": 11,                  # Pangu spatial height dimension
                "pangu_spatial_W": 11,                  # Pangu spatial width dimension
                "pangu1_time_dim": 2,                   # Pangu1 time dimension
                "pangu1_height_dim": 5,                 # Pangu1 height dimension
                "pangu1_features_dim": 13,              # Pangu1 features dimension
                "pangu2_time_dim": 2,                   # Pangu2 time dimension
                "pangu2_features_dim": 4,               # Pangu2 features dimension
                "pangu_output_ratio": 0.1               # Ratio of hidden_size for pangu features
            }
        }
    }
    
    # Define dimensions based on config
    batch_size = 32
    seq_length = 36
    num_turbines = len(config["turbine_indices"])
    num_features = len(config["feature_indices"])
    
    # Create the model instance
    model = GRCMI_WPF(config=config)
    
    # Dummy input data
    # Primary modality: [batch_size, seq_length, turbines, features]
    dummy_input = torch.rand(batch_size, seq_length, num_turbines, num_features)
    
    # Pangu data
    # pangu1: [bs, time_p1, height_p1, features_p1, H, W]
    pangu1 = torch.rand(batch_size, 2, 5, 13, 11, 11)
    # pangu2: [bs, time_p2, features_p2, H, W]
    pangu2 = torch.rand(batch_size, 2, 4, 11, 11)
    
    # Forward pass through the model
    output = model(dummy_input, pangu1, pangu2)
    print("Output shape:", output.shape)  # Expected: [32, 12]
    print("Model parameters:")
    print(f"  Hidden size: {config['model']['params']['hidden_size']}")
    print(f"  Pangu output ratio: {config['model']['params']['pangu_output_ratio']}")
    print(f"  Fusion weight: {config['model']['params']['fusion_weight']}")
    print(f"  Num heads: {config['model']['params']['num_heads']}")
    print(f"  Dropout: {config['model']['params']['dropout']}")
    
    # Dynamically adjust fusion weight
    model.set_fusion_weight(0.3)  # Now favor pangu modalities
    output_adjusted = model(dummy_input, pangu1, pangu2)
    print("Output shape (adjusted fusion):", output_adjusted.shape)  # Expected: [32, 12]