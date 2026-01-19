# PharmacologySE: A Pharmacology Knowledge Graph Embedding Framework

## Project Overview
PharmacologySE is a machine learning framework for predicting drug-drug interactions (DDIs) using knowledge graph embedding techniques. This framework integrates multiple drug features (smiles, target, enzyme) and employs advanced attention mechanisms to capture complex pharmacological relationships.

## Project Structure

```
PharmacologySE/
├── data/                     # Raw data files
│   ├── ddi_names.txt         # DDI type names
│   ├── ddi_smiles.txt        # Drug SMILES representations
│   ├── df_drugs.csv          # Drug information
│   ├── event.db              # SQLite database containing interaction events
│   ├── items.csv             # Additional drug items
│   └── items.pkl             # Pickled drug items
├── directory/                # Intermediate data files
│   └── environment_requirements.txt  # Full environment requirements
├── src/                     # Main code directory
│   ├── __pycache__/          # Python cache files
│   ├── bilstm_encoder.py     # BiLSTM encoder implementation
│   ├── config.py             # Model configuration
│   ├── cross_validation.py   # Cross-validation implementation
│   ├── data_sampling.py      # Data preprocessing and sampling
│   ├── evaluate.py           # Model evaluation metrics
│   ├── load_config.py        # Configuration loading utility
│   ├── main.py               # Main entry point
│   ├── model.py              # Model architecture implementation
│   ├── tools.py              # Utility functions
│   └── train.py              # Model training pipeline
├── output/                   # Model output files
│   ├── fusion_result.json    # Fusion results
│   ├── pred_score.txt        # Prediction scores
│   └── y_one_hot.txt         # One-hot encoded labels
├── results/                  # Experimental results
│   ├── allFolds*.csv         # Cross-validation results (all folds)
│   └── eachFold*.csv         # Cross-validation results (each fold)
├── test/                     # Test files
│   ├── test_data_sampling.py # Tests for data sampling
│   └── test_models.py        # Tests for model components
├── utils/                    # Utility functions
│   ├── bilstm_encoder.py     # BiLSTM encoder utility
│   ├── data_sampling.py      # Data sampling utility
│   ├── drug_list.py          # Drug list operations
│   └── get_target.py         # Target information retrieval
├── .idea/                    # IDE configuration (can be ignored)
├── paper.tex                 # Paper LaTeX source
└── requirements.txt          # Core dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA/NPU support (optional, for GPU acceleration)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PharmacologySE_260103/PharmacologySE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For full environment reproduction (optional):
```bash
pip install -r directory/environment_requirements.txt
```

## Usage

### Configuration
Modify the configuration parameters in `fold/config.py`:

```python
# Model parameters
seed = 2                        # Random seed for reproducibility
epo_num = 100                   # Number of epochs
batch_size = 128                # Batch size
Att_n_heads = 4                 # Number of attention heads
drop_out_rating = 0.5           # Dropout rate
learn_rating = 1.0e-5           # Learning rate
feature_list = ["smile", "target", "enzyme"]  # Features to use

# Device settings
DEVICE_TYPE = 'npu'             # Options: 'npu', 'cuda', 'cpu'
CUDA_VISIBLE_DEVICES = '0'      # GPU device ID
```

### Running the Model

Execute the main script:

```bash
cd fold
python main.py
```

### Cross-Validation
The model automatically performs 5-fold cross-validation. Results are saved in the `results/` directory.

## Reproducibility

To reproduce the results reported in the paper:

1. Ensure all dependencies are installed correctly
2. Use the exact configuration parameters from `fold/config.py`
3. Run the model with the provided data files
4. The results will be saved in the `results/` directory with filenames matching those in the paper

## Model Architecture

PharmacologySE employs a multi-head attention mechanism to integrate drug features:

1. **Feature Extraction**: Extracts features (smile, target, enzyme) from drug data
2. **Attention Mechanisms**: Three attention variants implemented:
   - MultiHeadSelfAttentionSem: Dot-product self-attention
   - MultiHeadCroAttentionSem: Dot-product cross-attention
   - MultiHeadSelfAttentionNod: Linear self-attention
3. **Prediction Layer**: Classifies DDI types based on the integrated features

## Data Description

### Input Data
- **Drug Features**: SMILES representations, target proteins, and enzyme information
- **Interaction Events**: Mechanism-action pairs extracted from pharmacological databases

### Data Preprocessing
1. Drug features are vectorized and normalized
2. Interaction events are categorized into DDI types
3. Features are concatenated for drug pairs

## Results Interpretation

### Output Files
- `allFolds*.csv`: Aggregated results across all cross-validation folds
- `eachFold*.csv`: Results for each individual fold
- `pred_score.txt`: Raw prediction scores for each sample

### Evaluation Metrics
The model reports standard classification metrics including accuracy, precision, recall, F1-score, and AUC-ROC.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our paper:

```
@article{PharmacologySE2024,
  title={PharmacologySE: A Knowledge Graph Embedding Framework for Drug-Drug Interaction Prediction},
  author={Song, et al.},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXXX-XXXX}
}
