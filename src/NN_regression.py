import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Constants ---
FEATURE_RANGE = (1e-6, 1.0) # Default normalization range

ARCHITECTURES = [
        {'name': 'Shallow', 'hidden_layers': 1, 'neurons_per_layer': 10, 'epochs': 1500, 'activation': 'relu'},
    ]

# --- Neural Network Definition ---
class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for regression."""
    def __init__(self, hidden_layers, neurons_per_layer, activation='relu'):
        super().__init__()
        layers = []
        act_fn = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}.get(activation, nn.ReLU())
        
        # Input layer (1 feature)
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(act_fn)
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(act_fn)
            
        # Output layer (1 output)
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# --- Data Handling ---
def load_and_normalize_data(data_dir: Path, feature_range: tuple = FEATURE_RANGE):
    """Loads all data splits and normalizes them based on training data stats."""
    datasets = {}
    for split in ["train", "validation", "test", "test_extra"]:
        try:
            datasets[split] = pd.read_csv(data_dir / f"{split}.csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {split}.csv not found in {data_dir}")

    print(f"Data loaded: " + ", ".join([f"{k}={len(v)}" for k, v in datasets.items()]))

    # Extract numpy arrays from training data for normalization params
    x_train_np = datasets['train']['x'].values
    y_train_np = datasets['train']['y'].values

    # Calculate normalization parameters from training data ONLY
    x_min, x_max = np.min(x_train_np), np.max(x_train_np)
    y_min, y_max = np.min(y_train_np), np.max(y_train_np)
    x_range = x_max - x_min + 1e-8 # Epsilon for stability
    y_range = y_max - y_min + 1e-8 # Epsilon for stability
    target_min, target_max = feature_range
    target_range = target_max - target_min

    norm_params = {'x_min': x_min, 'x_range': x_range,
                   'y_min': y_min, 'y_range': y_range,
                   'feature_range': feature_range}

    # Normalize and store tensors and original data
    norm_tensors = {}
    orig_data = {}

    for split, df in datasets.items():
        x_np = df['x'].values
        y_np = df['y'].values
        orig_data[f'x_{split}'] = x_np
        orig_data[f'y_{split}'] = y_np

        # Normalize x
        x_norm_np = target_min + target_range * (x_np - x_min) / x_range
        norm_tensors[f'x_{split}'] = torch.FloatTensor(x_norm_np.reshape(-1, 1))

        # Normalize y only for train and validation (used in loss calculation)
        if split in ['train', 'validation']:
            y_norm_np = target_min + target_range * (y_np - y_min) / y_range
            norm_tensors[f'y_{split}'] = torch.FloatTensor(y_norm_np.reshape(-1, 1))

    return norm_tensors, orig_data, norm_params

def denormalize_y(y_norm_tensor: torch.Tensor, norm_params: dict) -> torch.Tensor:
    """Denormalizes y tensor using stored parameters."""
    y_min, y_range = norm_params['y_min'], norm_params['y_range']
    target_min, target_max = norm_params['feature_range']
    target_range = target_max - target_min
    # Formula derived from: y_norm = target_min + target_range * (y_orig - y_min) / y_range
    y_denorm = y_min + (y_norm_tensor - target_min) * y_range / target_range
    return y_denorm

# --- Model Training and Evaluation ---
def calculate_denormalized_mse(model: nn.Module, x_tensor: torch.Tensor, y_orig_np: np.ndarray, norm_params: dict) -> float:
    """Calculates Mean Squared Error on the original data scale."""
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_tensor)
        y_pred_denorm = denormalize_y(y_pred_norm, norm_params)
        y_pred_denorm_np = y_pred_denorm.detach().numpy()
    mse = np.mean((y_pred_denorm_np.flatten() - y_orig_np.flatten())**2)
    return mse

def train_model(model: nn.Module, norm_tensors: dict, norm_params: dict, orig_data: dict, arch_config: dict):
    """Trains the model and returns final denormalized errors."""
    epochs = arch_config.get('epochs', 1000)
    batch_size = arch_config.get('batch_size', 32)
    lr = arch_config.get('lr', 0.001)

    train_dataset = TensorDataset(norm_tensors['x_train'], norm_tensors['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Training for {epochs} epochs (Batch Size: {batch_size}, LR: {lr})...")
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = criterion(torch.log(outputs), torch.log(targets))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Optional: Print progress (e.g., validation loss) periodically
        if (epoch + 1) % 200 == 0 or epoch == epochs - 1:
             model.eval()
             with torch.no_grad():
                 val_outputs = model(norm_tensors['x_validation'])
                 val_loss = criterion(val_outputs, norm_tensors['y_validation']).item()
             print(f'  Epoch {epoch+1}/{epochs}, Val Loss (Norm): {val_loss:.6f}')

    # Calculate final denormalized errors on all sets
    errors = {}
    for split in ["train", "validation", "test", "test_extra"]:
        errors[split] = calculate_denormalized_mse(
            model, norm_tensors[f'x_{split}'], orig_data[f'y_{split}'], norm_params
        )
    print(f"  Final Denorm MSE -> " + ", ".join([f"{k.capitalize()}: {v:.4f}" for k, v in errors.items()]))
    return errors

# --- Plotting ---
def plot_regression(orig_data: dict, model: nn.Module, norm_params: dict, arch_name: str, output_dir: Path, padding_factor=0.1):
    """Plots original data points and the denormalized model prediction line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original data points
    colors = {'train': 'k*', 'validation': 'bs', 'test': 'ro', 'test_extra': 'm^'}
    for split, marker in colors.items():
        ax.plot(orig_data[f'x_{split}'], orig_data[f'y_{split}'], marker, label=f'{split.capitalize()} data', markersize=5)

    # Create range for prediction line based on all data
    x_all = np.concatenate([orig_data[f'x_{split}'] for split in colors])
    y_all = np.concatenate([orig_data[f'y_{split}'] for split in colors])
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    x_range = x_max - x_min + 1e-8
    y_range = y_max - y_min + 1e-8
    x_pad = x_range * padding_factor
    y_pad = y_range * padding_factor

    x_plot_orig = np.linspace(x_min - x_pad, x_max + x_pad, 500)
    # Normalize plot range for model input
    target_min, target_max = norm_params['feature_range']
    x_plot_norm = target_min + (target_max - target_min) * (x_plot_orig - norm_params['x_min']) / norm_params['x_range']
    x_plot_tensor = torch.FloatTensor(x_plot_norm.reshape(-1, 1))

    # Get model predictions and denormalize
    model.eval()
    with torch.no_grad():
        y_plot_norm = model(x_plot_tensor)
    y_plot_denorm = denormalize_y(y_plot_norm, norm_params).numpy()

    ax.plot(x_plot_orig, y_plot_denorm, 'g-', linewidth=2, label=f'Model: {arch_name}')

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel('x (Original Scale)')
    ax.set_ylabel('y (Original Scale)')
    ax.set_title(f'NN Regression Results ({arch_name})')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_filename = output_dir / f'nn_regression_{arch_name.replace(" ", "_").lower()}.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"  Saved regression plot: {plot_filename}")
    plt.close(fig)

def plot_error_summary(results: list, output_dir: Path):
    """Plots final denormalized errors vs. model complexity."""
    if not results: return

    results.sort(key=lambda r: r['params']) # Sort by parameter count

    names = [f"{r['name']}\n({r['params']:,} params)" for r in results]
    params = [r['params'] for r in results]
    errors = {split: np.log10([r['errors'][split] for r in results]) for split in results[0]['errors']}

    _, ax = plt.subplots(figsize=(max(10, len(results) * 1.5), 7)) # Adjust width based on # models
    markers = {'train': 'ko-', 'validation': 'bs-', 'test': 'rx-', 'test_extra': 'm^-'}
    for split, marker in markers.items():
        ax.plot(params, errors[split], marker, label=f'Final {split.capitalize()} Error (log10 MSE)', markersize=6)

    ax.set_xticks(params)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Model Architecture and Parameter Count')
    ax.set_ylabel('log10(Final MSE - Denormalized)')
    ax.set_title('Final Denormalized Errors vs. Model Complexity')
    ax.legend()
    ax.grid(visible=True, which='major', axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'nn_error_vs_params_summary.png', dpi=300)
    plt.close()

def print_summary_table(results: list):
    """Prints a summary table of the results."""
    if not results: return

    print("\n--- Results Summary ---")
    # Dynamically create header based on error keys
    error_keys = list(results[0]['errors'].keys())
    header_fmt = "{:<15} {:<15} " + " ".join(["{:<18}"] * len(error_keys))
    header = header_fmt.format("Architecture", "Parameters", *[f"{k.capitalize()} MSE" for k in error_keys])
    print(header)
    print("-" * len(header))

    row_fmt = "{:<15} {:<15,} " + " ".join(["{:<18.4f}"] * len(error_keys))
    results.sort(key=lambda r: r['name']) # Sort alphabetically for table
    for r in results:
        print(row_fmt.format(r['name'], r['params'], *[r['errors'][k] for k in error_keys]))
    print("-" * len(header))

# --- Main Execution Logic ---
def run(args):
    """Main function to orchestrate loading, training, evaluation, and plotting."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load and prepare data
    norm_tensors, orig_data, norm_params = load_and_normalize_data(Path(args.data_dir), feature_range=(args.norm_min, 1.0))

    results = []
    for arch_config in ARCHITECTURES:
        arch_name = arch_config['name']
        print(f"\n--- Processing Architecture: {arch_name} ---")

        # Create model instance
        model = NeuralNetwork(
            arch_config['hidden_layers'],
            arch_config['neurons_per_layer'],
            arch_config.get('activation', 'relu')
        )
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {num_params:,}")

        # Train model and get final errors
        final_errors = train_model(model, norm_tensors, norm_params, orig_data, arch_config)

        # Store results
        results.append({
            'name': arch_name,
            'params': num_params,
            'errors': final_errors
            # Optionally store the trained model: 'model': model
        })

        # Plot individual regression fit
        plot_regression(orig_data, model, norm_params, arch_name, output_dir)

    # Generate summary plots and table
    plot_error_summary(results, output_dir)
    print_summary_table(results)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate NN regression models.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train/validation/test/test_extra.csv files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results and plots.")
    parser.add_argument("--norm_min", type=float, default=FEATURE_RANGE[0], help="Minimum value for Min-Max normalization range [norm_min, 1.0].")

    args = parser.parse_args()
    run(args)
 