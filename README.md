# TLAG-LSTM
TLAG-LSTM: Topologically Layer Adaptive Growing LSTM
Overview
TLAG-LSTM (Topologically Layer Adaptive Growing LSTM) is a novel deep learning model for streamflow prediction that can dynamically adjust its network architecture during training. Unlike traditional LSTM models with fixed layer counts, TLAG-LSTM uses topological derivative theory to automatically determine:

1. When to add a new layer
2. Where to insert the new layer in the network
3. How to initialize the new layer's weights
This adaptive approach helps the model find an optimal network depth for specific hydrological datasets without manual tuning or expensive hyperparameter optimization.

Key Features
1. Dynamic Network Growth: Automatically adds layers during training when beneficial
2. Optimal Layer Placement: Uses topological derivatives to determine the best position for new layers
3. Eigenvector-Based Initialization: Initializes new layers using eigenvectors of the Hessian matrix
4. Residual Connections: Employs residual connections in new layers to maintain network stability
5. Backtracking Strategy: Dynamically determines initialization scale for smooth network growth

Installation Requirements
pandas
numpy
scikit-learn
torch
scipy
matplotlib

Usage
Basic Usage
To use the TLAG-LSTM model, first import the necessary modules and create an instance of the AdaptiveLSTMModel class:

from TLAG-LSTM import AdaptiveLSTMModel, train_with_adaptation

Create model
model = AdaptiveLSTMModel(
input_size=precip.shape[1], # Number of precipitation stations
hidden_size=128, # Hidden dimension
initial_num_layers=3, # Start with 3 LSTM layers
dropout_between_layers=0.2 # Dropout between layers
)

Then train the model using the train_with_adaptation function:

Train with adaptation
model, architecture_history, training_losses = train_with_adaptation(
model=model,
train_loader=train_loader,
val_loader=val_loader,
scaler_flow=scaler_flow,
device=device,
max_layers=10, # Maximum allowed layers
patience=150, # Early stopping patience
learning_rate=0.0001, # Learning rate
topo_derivative_threshold=0.01 # Threshold for adding layers
)

Main Parameters
Model Creation Parameters:
input_size: Number of input features (precipitation stations)
hidden_size: Dimension of LSTM hidden state
initial_num_layers: Initial number of LSTM layers to start with
dropout_between_layers: Dropout rate between LSTM layers
num_power_iterations: Number of iterations for power method eigenvalue estimation
Training Parameters:
max_layers: Maximum allowed number of layers
patience: Early stopping patience value
learning_rate: Learning rate for optimizer
moving_avg_window: Window size for validation performance moving average
topo_derivative_threshold: Threshold above which a new layer is added
max_epochs_per_layer: Maximum training epochs per layer structure

Model Architecture
TLAG-LSTM consists of:
1. An input layer that takes precipitation data from multiple stations
2. A dynamically growing stack of LSTM layers
3. A fully connected output layer that produces streamflow predictions
The model starts with a small number of LSTM layers and progressively adds more layers during training based on topological derivative calculations.

How It Works
1. Training Phase:
(1)The model trains with the current architecture until validation performance stabilizes;
(2)Topological derivatives are calculated for each layer;
(3)If the maximum derivative exceeds the threshold, a new layer is added.

2. Layer Addition:
(1)The new layer is inserted after the layer with the highest topological derivative;
(2)The layer is initialized using the eigenvector of the Hessian matrix;
(3)A backtracking strategy determines the optimal initialization scale.

3. Convergence:
(1)The process continues until all layers' topological derivatives fall below the threshold;
(2)The final model represents the optimal network depth for the given dataset.

Example Output
The model produces several output files:
1. lstm_architecture_evolution.json: Records the evolution of network architecture
2. lstm_evolution.png: Visualization of architecture evolution and performance
3. lstm_loss.png: Training loss evolution plot
4. lstm_prediction.png: Prediction results visualization
5. adaptive_lstm_model.pth: Saved final model
6. adaptive_lstm_architecture.txt: Information about the final model architecture
7. topological_derivatives_history.xlsx: History of topological derivative calculations

Citation
If you use TLAG-LSTM in your research, please cite:
Sha, J., Zheng, J., Yue, F., Li, X., & Liu, X. (2025). A Novel Layer Adaptive Growing LSTM with Dynamic Neural Network Architecture Guided by Topological Derivative for Streamflow Prediction.

License
This project is licensed under the MIT License.
