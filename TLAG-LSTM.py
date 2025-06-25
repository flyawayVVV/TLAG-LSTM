import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import json
import math
import copy
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Calculate NSE metric
def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - num / den

# Create temporal sequences
def create_sequences(precip, flow, n_days):
    X, y = [], []
    for i in range(n_days - 1, len(flow)):
        X.append(precip[i - n_days + 1 : i + 1])
        y.append(flow[i])
    return np.stack(X), np.array(y).reshape(-1, 1)

# Topological Derivative Module - Modified Hamiltonian Calculation
class TopologicalDerivative:
    def __init__(self, num_power_iterations=10):
        self.derivatives = {}
        self.eigenvectors = {}
        self.max_eigenvalues = {}
        self.current_H = None  # Store current layer's Hamiltonian for Hessian-vector
        self.num_power_iterations = num_power_iterations

    def compute_hamiltonian(self, model, x, y, criterion):
        training_state = model.training
        model.train()

        derivatives = {}
        eigenvectors = {}
        max_eigenvalues = {}

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            # 1) Forward + Backward to get p_T = dL/dy_pred
            model.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            p_T = torch.autograd.grad(loss, y_pred, create_graph=True)[0]  # [batch,1]

            # 2) Register hooks to collect outputs from each LSTM layer
            activations = {}
            hooks = []
            def get_activation(name):
                def hook(mod, inp, out):
                    # For standard nn.LSTM, out=(seq_out, (h_n,c_n))
                    # For our custom ResidualLSTMCell, out is seq_out
                    seq_out = out[0] if isinstance(out, tuple) else out
                    activations[name] = seq_out
                return hook
            for i, lstm in enumerate(model.lstm_layers):
                hooks.append(lstm.register_forward_hook(get_activation(str(i))))
            _ = model(x)
            for h in hooks:
                h.remove()

            # 3) Build Hamiltonian for each layer output and estimate max Hessian eigenvalue using power iteration
            for i, lstm in enumerate(model.lstm_layers):
                layer_idx = str(i)
                layer_params = list(lstm.parameters())
                if not layer_params:
                    continue

                # Get last hidden state of current layer
                out = activations[layer_idx]           # [batch, seq_len, hidden]
                h_last = out[:, -1, :]                 # [batch, hidden]
                # Hamiltonian H = sum(p_T * h_last)
                H_l = (p_T.detach() * h_last).sum()
                self.current_H = H_l                   # Store for Hessian-vector use

                total = sum(p.numel() for p in layer_params)
                # Power iteration for max eigenvalue & vector
                max_eig, vec = self._power_iteration(layer_params, total, 10)
                derivatives[layer_idx] = max_eig
                eigenvectors[layer_idx] = vec
                max_eigenvalues[layer_idx] = max_eig
                print(f"Calculated topological derivative (max eigenvalue) for layer {layer_idx}: {max_eig:.10f}")

            if not derivatives:
                print("Warning: No valid parameters found, using default values")
                derivatives["0"] = 0.01
                eigenvectors["0"] = torch.ones(1)
                max_eigenvalues["0"] = 0.01

            self.derivatives = derivatives
            self.eigenvectors = eigenvectors
            self.max_eigenvalues = max_eigenvalues
            return derivatives, eigenvectors, max_eigenvalues

        finally:
            model.train(training_state)
            torch.backends.cudnn.enabled = cudnn_enabled

    def _power_iteration(self, layer_params, dim, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.num_power_iterations
        # Random initialization
        device = layer_params[0].device
        v = torch.randn(dim, device=device)
        v = v / v.norm()
        for _ in range(num_iterations):
            v = self._hessian_vector_product(layer_params, v)
            norm = v.norm()
            if norm > 0:
                v = v / norm
        eig = self._compute_rayleigh(layer_params, v)
        return eig.item(), v

    def _hessian_vector_product(self, layer_params, v):
        # First-order gradient
        grads = torch.autograd.grad(self.current_H, layer_params, create_graph=True)
        # grad · v
        gv = torch.zeros(1, device=v.device)
        offset = 0
        for g in grads:
            flat = g.reshape(-1)
            n = flat.numel()
            gv = gv + (flat * v[offset:offset+n]).sum()
            offset += n
        # Second-order derivative
        hv = torch.autograd.grad(gv, layer_params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv])

    def _compute_rayleigh(self, layer_params, v):
        hv = self._hessian_vector_product(layer_params, v)
        return (v * hv).sum() / (v * v).sum()

    def get_insertion_location(self):
        max_layer, max_val, max_vec = None, -float('inf'), None
        for idx, val in self.derivatives.items():
            fv = float(val)
            if fv > max_val:
                max_val, max_layer, max_vec = fv, idx, self.eigenvectors[idx]
        if max_layer is None:
            return "0", 0.0, None, 0
        return max_layer, max_val, max_vec, int(max_layer)

# Residual LSTM Cell (Message Passing Layer)
class ResidualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, eigenvector=None, init_scale=0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        if eigenvector is not None:
            self._initialize_with_eigenvector(eigenvector, init_scale)
        else:
            for param in self.lstm.parameters():
                nn.init.zeros_(param)

    def _initialize_with_eigenvector(self, eigenvector, init_scale):
        # Initialize weights and biases with eigenvector
        offset = 0
        total_params = sum(p.numel() for p in self.lstm.parameters())
        if eigenvector.numel() != total_params:
            print(f"Warning: Eigenvector dimension {eigenvector.numel()} does not match required parameter dimension {total_params}, attempting auto-expansion")
            # Auto-expand vector dimension (using repetition or linear interpolation)
            repeats = total_params // eigenvector.numel()
            remainder = total_params % eigenvector.numel()
            # Repeat integer part
            expanded = eigenvector.repeat(repeats)
            # Add remainder
            if remainder > 0:
                expanded = torch.cat([expanded, eigenvector[:remainder]])
            eigenvector = expanded
            print(f"Expanded eigenvector to {eigenvector.numel()}")
        for name, param in self.lstm.named_parameters():
            numel = param.numel()
            vec_slice = eigenvector[offset:offset+numel].view_as(param)
            param.data.copy_(init_scale * vec_slice)
            offset += numel

    def forward(self, x):
        out, _ = self.lstm(x)
        return x + out  # Residual connection, no extra scale parameter needed
    
# Use BackTracking strategy from paper appendix E.1 to automatically determine initialization scale epsilon
def backtracking_initialization(model, position, eigenvector, device,
                                 base_scale=0.1, min_scale=0.00001, tolerance=0.01,
                                 max_attempts=20):
    print(f"Attempting to dynamically determine initialization scale epsilon after position {position}")
    criterion = nn.MSELoss()

    # Get a batch of training data for loss evaluation
    sample_x, sample_y = next(iter(model.sample_loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)

    # Original loss
    model.eval()
    with torch.no_grad():
        pred = model(sample_x)
        loss_old = criterion(pred, sample_y).item()

    epsilon = base_scale

    for attempt in range(max_attempts):
        # Test new layer insertion on original model
        new_cell = ResidualLSTMCell(
            input_size=model.hidden_size,
            hidden_size=model.hidden_size,
            eigenvector=eigenvector,
            init_scale=epsilon
        ).to(device)

        model.lstm_layers.insert(position + 1, new_cell)
        model.dropouts.insert(position + 1, nn.Dropout(model.dropout_between_layers))

        # Test loss
        model.eval()
        with torch.no_grad():
            pred_new = model(sample_x)
            loss_new = criterion(pred_new, sample_y).item()

        # Immediately remove newly inserted layer (preserving original model structure)
        model.lstm_layers.pop(position + 1)
        model.dropouts.pop(position + 1)

        # Evaluate
        if loss_new <= loss_old * (1 + tolerance):
            print(f"BackTracking successful: epsilon={epsilon:.5f}, loss_old={loss_old:.10f}, loss_new={loss_new:.10f}")
            return epsilon
        else:
            print(f"BackTracking failed: epsilon={epsilon:.5f}, loss_old={loss_old:.10f}, loss_new={loss_new:.10f} exceeds tolerance")
            epsilon /= 2
            if epsilon < min_scale:
                print("epsilon too small, terminating backtracking, using minimum value")
                epsilon = min_scale
                break

    return epsilon

# Adaptive LSTM Model
class AdaptiveLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, initial_num_layers, dropout_between_layers=0.0, num_power_iterations=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_between_layers = dropout_between_layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i==0 else hidden_size, hidden_size, batch_first=True)
            for i in range(initial_num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_between_layers)
            for _ in range(initial_num_layers)
        ])
        self.fc = nn.Linear(hidden_size, 1)
        self.topological_derivative = TopologicalDerivative(num_power_iterations=num_power_iterations)

    def forward(self, x):
        for lstm, do in zip(self.lstm_layers, self.dropouts):
            out = lstm(x)
            # For standard nn.LSTM, out will be (seq_out, (h_n,c_n))
            # For ResidualLSTMCell, out is just Tensor
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
            x = do(x)
        # Last time step
        out_last = x[:, -1, :]
        return self.fc(out_last)

    def add_layer(self, position=None, eigenvector=None, initialization_scale=0.01):
        if position is None or position >= len(self.lstm_layers):
            position = len(self.lstm_layers)
        print(f"Adding new LSTM layer after position {position}")

        # Determine input dimension for new layer
        input_dim = self.hidden_size  # Input dimension is hidden_size

        if eigenvector is None or eigenvector.numel() == 0:
            print("Warning: No valid eigenvector provided, using default zero initialization")
            eigenvector = None
        
        new_cell = ResidualLSTMCell(
            input_dim,                        # <-- Dynamically determine input dimension
            self.hidden_size, 
            eigenvector=eigenvector,          # <-- Use eigenvector for initialization
            init_scale=initialization_scale
        )
    
        new_dropout = nn.Dropout(self.dropout_between_layers)
        new_cell.to(next(self.parameters()).device)
        new_dropout.to(next(self.parameters()).device)
        self.lstm_layers.insert(position+1, new_cell)
        self.dropouts.insert(position+1, new_dropout)
        print(f"Added new LSTM layer, current total layers: {len(self.lstm_layers)}")
        return new_cell

    def print_architecture(self):
        print("\nCurrent Network Architecture:")
        print(f"Input Dimension: {self.input_size}")
        print(f"Hidden Dimension: {self.hidden_size}")
        print(f"Number of LSTM Layers: {len(self.lstm_layers)}")
        print(f"Dropout Between Layers: {self.dropout_between_layers}")
        print(f"Output Layer: Linear Layer ({self.hidden_size} -> 1)")

# Evaluation function
def evaluate(model, loader, scaler_flow, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).cpu().numpy()
            ys.append(yb.cpu().numpy())
            ps.append(pred)
    y_true = np.vstack(ys)
    y_pred = np.vstack(ps)
    y_true_orig = scaler_flow.inverse_transform(y_true)
    y_pred_orig = scaler_flow.inverse_transform(y_pred)
    if np.all(y_true_orig==y_true_orig[0]) or np.all(y_pred_orig==y_pred_orig[0]):
        print("Warning: Constant array detected, setting Pearson correlation to 0")
        r = 0.0
    else:
        r = pearsonr(y_true_orig.flatten(), y_pred_orig.flatten())[0]
    n = nse(y_true_orig.flatten(), y_pred_orig.flatten())
    return r, n, y_true_orig, y_pred_orig

# Adaptive training using topological derivative
def train_with_adaptation(model, train_loader, val_loader, scaler_flow, device,
                          max_layers=float('inf'), patience=10, 
                          learning_rate=0.0005, moving_avg_window=5,
                          topo_derivative_threshold=0, max_epochs_per_layer=1000):
    criterion = nn.MSELoss()
    architecture_history = [{
        'epoch':0,'num_layers':len(model.lstm_layers),
        'validation_nse':0.0,'avg_validation_nse':0.0,'layer_added':False
    }]
    
    # Create directory for intermediate models
    model_dir = "intermediate_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save all topological derivative calculation results
    topo_derivatives_history = []
    
    # Record training losses
    training_losses = []
    
    total_epoch = 0
    current_layer_count = len(model.lstm_layers)
    stop_training = False
    while not stop_training and current_layer_count <= max_layers:
        print(f"\nStarting training with current architecture: LSTM layers = {current_layer_count}")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        validation_nse_history, best_avg_val_nse = [], -float('inf')
        best_model_state, no_improve_count = None, 0
        epoch_in_layer = 0
        
        epoch_losses = []  # Training losses within current layer
        
        while no_improve_count < patience and epoch_in_layer < max_epochs_per_layer:
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                
            # Calculate average loss
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
                
            r_tr, n_tr, _, _ = evaluate(model, train_loader, scaler_flow, device)
            r_va, n_va, _, _ = evaluate(model, val_loader, scaler_flow, device)
            validation_nse_history.append(n_va)
            if len(validation_nse_history) > moving_avg_window:
                validation_nse_history.pop(0)
            avg_val_nse = sum(validation_nse_history)/len(validation_nse_history)
            total_epoch += 1; epoch_in_layer += 1
            print(f"Total Epoch {total_epoch:4d} | Layer Epoch {epoch_in_layer:3d} | "
                  f"Loss={avg_epoch_loss:.6f} | "
                  f"Train: Pearson={r_tr:.4f}, NSE={n_tr:.4f} | "
                  f"Valid: Pearson={r_va:.4f}, NSE={n_va:.4f} (Avg: {avg_val_nse:.4f}) | "
                  f"Layers: {len(model.lstm_layers)}")
            record = {'epoch':total_epoch,'num_layers':len(model.lstm_layers),
                      'validation_nse':n_va,'avg_validation_nse':avg_val_nse,
                      'layer_added':False, 'training_loss': avg_epoch_loss}
            architecture_history.append(record)
            training_losses.append(avg_epoch_loss)
            
            if len(validation_nse_history)==moving_avg_window and avg_val_nse>best_avg_val_nse:
                best_avg_val_nse, best_model_state = avg_val_nse, {
                    k:v.cpu().clone() for k,v in model.state_dict().items()
                }
                no_improve_count=0
                print(f"Found new best model for current architecture, average validation NSE: {best_avg_val_nse:.4f}")
                
                # Save best model for current layer structure
                model_path = os.path.join(model_dir, f"best_model_layers_{len(model.lstm_layers)}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model for current architecture ({len(model.lstm_layers)} layers) to: {model_path}")
            else:
                no_improve_count+=1
                print(f"No improvement in validation: {no_improve_count}/{patience}, current best moving average NSE: {best_avg_val_nse:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restored model with highest validation NSE for current architecture (NSE: {best_avg_val_nse:.4f})")
        
        if current_layer_count < max_layers:
            # Calculate topological derivative and decide layer addition
            try:
                sample_x, sample_y = next(iter(train_loader))
                sample_x, sample_y = sample_x.to(device), sample_y.to(device)
                derivs, vecs, _ = model.topological_derivative.compute_hamiltonian(
                    model, sample_x, sample_y, criterion)
                layer_name, max_deriv, max_vec, idx = model.topological_derivative.get_insertion_location()
                print("\nTopological derivative values (higher indicates better for adding new layer):")
                
                # Save topological derivative calculation results
                topo_derivative_record = {
                    'epoch': total_epoch,
                    'num_layers': len(model.lstm_layers),
                    'derivatives': {}
                }
                
                for l,d in derivs.items():
                    print(f"  Layer {l}: {d:.10f}")
                    topo_derivative_record['derivatives'][l] = d
                
                topo_derivatives_history.append(topo_derivative_record)
                
                if max_deriv > topo_derivative_threshold:
                    print(f"\nDecided to add new layer after layer {layer_name} (index {idx}), topological derivative value: {max_deriv:.10f}")
                    # Temporarily bind training data loader to model (for sampling)
                    model.sample_loader = train_loader
                    epsilon = backtracking_initialization(
                        model=model,
                        position=idx,
                        eigenvector=max_vec,
                        device=device,
                        tolerance=0
                    )                
                    model.add_layer(position=idx, eigenvector=max_vec, initialization_scale=epsilon)
                    
                    print(f"Layer insertion complete, using dynamically selected epsilon = {epsilon:.5f}")
                    current_layer_count = len(model.lstm_layers)
                    
                    # Mark new layer addition
                    architecture_history[-1]['layer_added'] = True
                else:
                    print(f"\nMaximum topological derivative value {max_deriv:.10f} below threshold {topo_derivative_threshold}")
                    print("Stopping layer addition, training complete")
                    stop_training = True
            except Exception as e:
                print(f"Error in topological derivative calculation or layer addition: {str(e)}")
                import traceback
                traceback.print_exc()
                stop_training = True
        else:
            print("Maximum layer count reached, stopping layer addition, using model with highest validation NSE, training complete")
            current_layer_count += 1    # To end while loop by triggering current_layer_count <= max_layers
    
    # Save topological derivative history to Excel
    if topo_derivatives_history:
        df_topo = []
        for record in topo_derivatives_history:
            row = {
                'epoch': record['epoch'],
                'num_layers': record['num_layers']
            }
            for layer_idx, deriv_value in record['derivatives'].items():
                row[f'layer_{layer_idx}_derivative'] = deriv_value
            df_topo.append(row)
        
        pd.DataFrame(df_topo).to_excel('topological_derivatives_history.xlsx', index=False)
        print("Saved topological derivative history to 'topological_derivatives_history.xlsx'")
    
    return model, architecture_history, training_losses

# Main function and hyperparameter definition section
def main():
    # ========== User-defined parameters ==========
    input_path  = "DemoData.xlsx"      # Input Excel file path
    output_path = "results-adaptive-lstm.xlsx"
    n_days      = 30                     # Number of precipitation days to use (including current day)
    hidden_size = 128                    # LSTM hidden layer dimension
    initial_num_layers = 3               # Initial number of LSTM layers
    max_layers  = 10                     # Maximum layer count limit
    lr          = 0.0001                 # Learning rate
    batch_size  = 128                    # Batch size
    patience    = 150                    # Early stopping patience value (consecutive epochs without improvement before considering layer training sufficient)
    moving_avg_window = 5                # Validation performance moving average window size
    dropout_between_layers = 0.2         # Dropout rate between layers
    topo_derivative_threshold = 0.01     # Topological derivative threshold, only add new layer if above this value
    max_epochs_per_layer = 10000         # Maximum training epochs limit per layer
    power_iteration_steps = 10           # Power iteration steps
    # ===================================

    seed = 42       # Ensure training results are reproducible for review
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Read data
    try:
        df = pd.read_excel(input_path)
        print(f"Successfully loaded {input_path}")
    except FileNotFoundError:
        print(f"Warning: {input_path} not found")
        return

    # Assumption: First 3 columns are year, month, day, fourth column is flow, fifth column onwards are multi-site precipitation
    df_dates = pd.to_datetime({
        'year':  df.iloc[:, 0].astype(int),
        'month': df.iloc[:, 1].astype(int),
        'day':   df.iloc[:, 2].astype(int)
    })
    flow = df.iloc[:, 3].values.reshape(-1, 1).astype(float)
    precip = df.iloc[:, 4:].values.astype(float)

    # Check for NaN and print specific locations
    nan_p = np.argwhere(np.isnan(precip))
    for r, c in nan_p:
        print(f"NaN in precipitation at row {r}, column '{df.columns[4 + c]}'")
    nan_f = np.argwhere(np.isnan(flow))
    for r, _ in nan_f:
        print(f"NaN in flow at row {r}, column '{df.columns[3]}'")

    # Replace NaN values
    if len(nan_p) > 0:
        print(f"Found {len(nan_p)} NaN values in precipitation data, replacing with 0")
        precip = np.nan_to_num(precip, nan=0.0)
    
    if len(nan_f) > 0:
        print(f"Found {len(nan_f)} NaN values in flow data, replacing with 0")
        flow = np.nan_to_num(flow, nan=0.0)

    # 2. Normalization
    scaler_prec = MinMaxScaler()
    precip_norm = scaler_prec.fit_transform(precip)
    scaler_flow = MinMaxScaler()
    flow_norm   = scaler_flow.fit_transform(flow)

    # 3. Create temporal sequences
    X, y = create_sequences(precip_norm, flow_norm, n_days)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Corresponding sample dates (corresponding to y positions), keep as Series/DatetimeIndex
    sample_dates = df_dates[n_days-1:]

    # 4. Data split: 80% training+validation, 20% test (sequential, no random)
    N = len(X)
    train_val_end = int(N * 0.8)
    X_train_val, y_train_val = X[:train_val_end], y[:train_val_end]
    X_test, y_test = X[train_val_end:], y[train_val_end:]
    dates_test = sample_dates.iloc[train_val_end:]

    # Training set split again: 80% training, 20% validation (sequential, no random)
    tv_N = len(X_train_val)
    train_end = int(tv_N * 0.8)
    X_train, y_train = X_train_val[:train_end], y_train_val[:train_end]
    X_val, y_val     = X_train_val[train_end:], y_train_val[train_end:]

    print(f"Training set size: {len(X_train)} sequences")
    print(f"Validation set size: {len(X_val)} sequences")
    print(f"Test set size: {len(X_test)} sequences")

    # 5. DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False)

    # 6. Create adaptive LSTM model
    model = AdaptiveLSTMModel(
        input_size=precip.shape[1],
        hidden_size=hidden_size,
        initial_num_layers=initial_num_layers,
        dropout_between_layers=dropout_between_layers,
        num_power_iterations=power_iteration_steps
    ).to(device)

    print("Initial model architecture:")
    model.print_architecture()

    # 7. Train using topological derivative method for adaptation
    print("\nStarting training of LSTM model with topological derivative adaptation...")
    model, architecture_history, training_losses = train_with_adaptation(
        model,
        train_loader,
        val_loader,
        scaler_flow=scaler_flow,
        device=device,
        max_layers=max_layers,
        patience=patience,
        learning_rate=lr,
        moving_avg_window=moving_avg_window,
        topo_derivative_threshold=topo_derivative_threshold,
        max_epochs_per_layer=max_epochs_per_layer
    )

    # 8. Save architecture evolution history
    try:
        with open('lstm_architecture_evolution.json', 'w') as f:
            # Convert non-serializable types
            serializable_history = []
            for record in architecture_history:
                serializable_record = {}
                for k, v in record.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        serializable_record[k] = v
                    else:
                        serializable_record[k] = str(v)
                serializable_history.append(serializable_record)

            json.dump(serializable_history, f, indent=4)
            print("Architecture evolution history saved to 'lstm_architecture_evolution.json'")
    except Exception as e:
        print(f"Error saving architecture evolution history: {str(e)}")

    # 9. Visualize architecture evolution and performance changes
    try:
        # Create figure with two subplots (shared x-axis)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        epochs = [rec['epoch'] for rec in architecture_history]
        nse_values = [rec.get('validation_nse', 0.0) for rec in architecture_history]
        avg_nse_values = [rec.get('avg_validation_nse', 0.0) for rec in architecture_history]
        layers = [rec['num_layers'] for rec in architecture_history]
        
        # NSE curve (upper subplot)
        color = 'tab:blue'
        ax1.set_ylabel('Validation NSE', color=color)
        ax1.plot(epochs, nse_values, color=color, label='Single NSE', alpha=0.3)
        ax1.plot(epochs, avg_nse_values, color='tab:red', label='Moving Avg NSE')
        ax1.tick_params(axis='y', labelcolor=color)

        # Mark layer addition positions
        for i, rec in enumerate(architecture_history):
            if rec.get('layer_added', False):
                ax1.axvline(x=rec['epoch'], color='r', linestyle='--', alpha=0.5)
                # Ensure proper text position
                y_values = [v for v in nse_values if not math.isnan(v) and v != 0]
                if y_values:
                    y_pos = min(y_values) + 0.1 * (max(y_values) - min(y_values))
                    ax1.text(rec['epoch'], y_pos, f"Add Layer",
                             rotation=90, color='r', verticalalignment='center')

        # Layer count curve
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Number of Layers', color=color)
        ax2.plot(epochs, layers, color=color, label='Layers')
        ax2.tick_params(axis='y', labelcolor=color)

        # Merge legends and place above plot
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        ax1.set_title('LSTM Architecture Evolution via Topological Derivative')
        
        # Training Loss curve (lower subplot)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Training Loss')
        ax3.plot(epochs[1:], training_losses, color='tab:orange')
        
        # Mark layer addition positions
        for i, rec in enumerate(architecture_history):
            if rec.get('layer_added', False):
                ax3.axvline(x=rec['epoch'], color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save high-resolution image
        plt.savefig('lstm_evolution.png', dpi=1200)
        print("\nArchitecture evolution plot saved as 'lstm_evolution.png' (1200 dpi)")
        
        # Separately save training Loss plot
        plt.figure(figsize=(12, 6))
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.plot(epochs[1:], training_losses, color='tab:orange')
        
        # Mark layer addition positions
        for i, rec in enumerate(architecture_history):
            if rec.get('layer_added', False):
                plt.axvline(x=rec['epoch'], color='r', linestyle='--', alpha=0.5)
                plt.text(rec['epoch'], max(training_losses), f"Add Layer",
                        rotation=90, color='r', verticalalignment='top')
                
        plt.title('Training Loss Evolution')
        plt.tight_layout()
        plt.savefig('lstm_loss.png', dpi=1200)
        print("Training Loss evolution plot saved as 'lstm_loss.png' (1200 dpi)")
    
    except ImportError:
        print("Warning: matplotlib not installed, cannot generate visualization plots")
    except Exception as e:
        print(f"Error generating visualization plots: {str(e)}")
        import traceback
        traceback.print_exc()

    # 10. Test set evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())
    y_true = scaler_flow.inverse_transform(np.vstack(all_trues)).flatten()
    y_pred = scaler_flow.inverse_transform(np.vstack(all_preds)).flatten()

    # Calculate evaluation metrics
    r_te = pearsonr(y_true, y_pred)[0]
    n_te = nse(y_true, y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    nmae  = mae / (y_true.max() - y_true.min())
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nFinal Test Results:")
    print(f"  Pearson Correlation:     {r_te:.4f}")
    print(f"  NSE:                     {n_te:.4f}")
    print(f"  NMAE:                    {nmae:.4f}")
    print(f"  MAPE:                    {mape:.2f}%")
    print("Note that since the Demo uses Min-Max normalized data, the MAPE is not the same as the MAPE value calculated using the original values")
    print(f"  Final Number of Layers:  {len(model.lstm_layers)}")

    # 11. Visualize prediction results
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue', linewidth=2)
        plt.plot(y_pred, label='Predicted', color='red', linewidth=2)
        plt.xlabel('Sample Index')
        plt.ylabel('Flow')
        plt.title('Flow Prediction Results')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('lstm_prediction.png', dpi=1200)
        print("Prediction results plot saved as 'lstm_prediction.png' (1200 dpi)")
    except Exception as e:
        print(f"Error saving prediction plot: {str(e)}")

    # 12. Save trained model
    torch.save(model.state_dict(), "adaptive_lstm_model.pth")
    
    # Save model architecture information
    with open('adaptive_lstm_architecture.txt', 'w') as f:
        f.write("====== Optimal LSTM Structure Determined by Topological Derivative ======\n")
        f.write(f"Input Dimension: {model.input_size}\n")
        f.write(f"Hidden Dimension: {model.hidden_size}\n")
        f.write(f"Number of LSTM Layers: {len(model.lstm_layers)}\n")
        f.write(f"Dropout Between Layers: {model.dropout_between_layers}\n")
        f.write(f"Topological Derivative Threshold: {topo_derivative_threshold}\n\n")
    
    print("Final model saved to 'adaptive_lstm_model.pth'")
    print("Model architecture information saved to 'adaptive_lstm_architecture.txt'")

    # 13. Save test results to Excel, including pure "YYYY-MM-DD" format dates
    out_df = pd.DataFrame({
        "Date":      dates_test.dt.strftime('%Y-%m-%d'),
        "True":      y_true,
        "Predicted": y_pred
    })
    out_df.to_excel(output_path, index=False)
    print(f"Test results saved to {output_path}")
    

        

if __name__ == "__main__":
    main()
