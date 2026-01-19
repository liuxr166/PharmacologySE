# -*- coding: utf-8 -*-
"""
Configuration loading utility for PharmacologySE framework.

This module provides functions to load and validate configuration settings
from a config.py file, and set up the appropriate computing device.
"""
import os
import torch
import importlib.util
from pathlib import Path

try:
    import torch_npu
except ImportError:
    torch_npu = None


def load_config(config_path='config.py'):
    """
    Load configuration from a Python file.
    
    Args:
        config_path (str): Path to the configuration file. Defaults to 'config.py'.
        
    Returns:
        tuple:
            - params (dict): Dictionary containing all configuration parameters
            - device (torch.device): Selected computing device (npu, cuda, or cpu)
            - results_dir (str): Directory path for saving results
    
    Raises:
        FileNotFoundError: If the configuration file is not found
        ImportError: If there's an error importing the configuration file
    """
    # Ensure configuration file exists
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Convert to absolute path for better error messages
    config_path = config_path.absolute()
    
    try:
        # Dynamically import the configuration file
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except Exception as e:
        raise ImportError(f"Error loading configuration file {config_path}: {str(e)}")
    
    # Set CUDA device environment variable
    cuda_devices = getattr(config, 'CUDA_VISIBLE_DEVICES', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    
    # Determine the appropriate computing device
    device_type = getattr(config, 'DEVICE_TYPE', 'cpu')
    device = _get_device(device_type)
    
    # Collect all configuration parameters
    exclude_attrs = [
        'CUDA_VISIBLE_DEVICES', 'DEVICE_TYPE', 
        '__name__', '__doc__', '__package__',
        '__loader__', '__spec__', '__file__', 
        '__cached__', '__builtins__'
    ]
    
    params = {}
    for attr_name in dir(config):
        if not attr_name.startswith('_') and attr_name not in exclude_attrs:
            params[attr_name] = getattr(config, attr_name)
    
    # Validate required parameters
    _validate_required_params(params)
    
    # Create results directory if it doesn't exist
    results_dir = getattr(config, 'file_path', './results/')
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration loaded successfully from {config_path}")
    print(f"Using device: {device}")
    print(f"Results will be saved to: {results_dir}")
    
    return params, device, str(results_dir)


def _get_device(device_type):
    """
    Get the appropriate torch device based on the specified device type and availability.
    
    Args:
        device_type (str): Device type ('npu', 'cuda', or 'cpu')
        
    Returns:
        torch.device: Selected computing device
    """
    if device_type == 'npu' and torch_npu and torch_npu.npu.is_available():
        device = torch.device("npu")
        print("Using NPU device for computation")
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for computation")
    else:
        device = torch.device("cpu")
        if device_type != 'cpu':
            print(f"Requested device type '{device_type}' not available, falling back to CPU")
        else:
            print("Using CPU for computation")
    
    return device


def _validate_required_params(params):
    """
    Validate that all required parameters are present in the configuration.
    
    Args:
        params (dict): Configuration parameters
        
    Raises:
        ValueError: If required parameters are missing
    """
    required_params = [
        'seed', 'epo_num', 'batch_size', 
        'Att_n_heads', 'cross_ver_tim', 
        'drop_out_rating', 'learn_rating',
        'weight_decay_rate', 'feature_list'
    ]
    
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Validate feature_list is a non-empty list
    if not isinstance(params['feature_list'], list) or len(params['feature_list']) == 0:
        raise ValueError("feature_list must be a non-empty list")
    
    print(f"All required parameters validated successfully")


if __name__ == '__main__':
    """Example usage and testing of the configuration loading function."""
    try:
        params, device, results_dir = load_config()
        
        # Display loaded parameters
        print("\n=== Loaded Configuration Parameters ===")
        for key, value in sorted(params.items()):
            print(f"{key}: {value}")
        
        print(f"\nDevice: {device}")
        print(f"Results Directory: {results_dir}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()