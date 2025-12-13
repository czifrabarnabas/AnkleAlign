# AnkleAlign

Ankle alignment classification using deep learning. This project classifies posterior foot/ankle photographs into three categories: **Pronation** (inward tilt), **Neutral**, and **Supination** (outward tilt).

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Czifra Barnabás
- **Aiming for +1 Mark**: No

### Solution Description

This project implements a deep learning solution for automatic ankle alignment classification from rear-view photographs of feet and ankles. The system can assist in early detection of foot and posture problems.

**Problem**: Manual assessment of ankle alignment is time-consuming and requires expert knowledge. Automated classification can provide quick screening assistance.

**Model Architecture**: We use a ResNet-18 convolutional neural network pretrained on ImageNet, with transfer learning to adapt it to our 3-class ankle alignment task. The final fully connected layer is replaced with a new layer outputting 3 classes.

**Training Methodology**:
- Transfer learning from ImageNet pretrained weights
- Data augmentation: random crop, horizontal flip, rotation (±15°), color jitter
- Optimizer: Adam with learning rate 1e-4
- Loss function: Cross-entropy
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping with patience of 10 epochs
- Stratified train/validation/test split (70/15/15)

**Results**: The model achieves approximately 70% validation accuracy and 73% test accuracy, with best performance on the Neutral class due to class imbalance in the dataset (965 total samples: 559 Neutral, 193 Pronation, 81 Supination).

### Data Preparation

The data is provided in Label Studio JSON consensus format. The raw data is located on a shared drive and should be mounted at `/data` when running the container.

**Data structure expected:**
```
/data/
└── anklealign/
    └── anklealign/
        ├── consensus/           # JSON annotation files
        │   ├── file1.json
        │   └── ...
        ├── internet_*/          # Internet-sourced images
        └── sajat_*/             # Self-collected images
```

The `01_data_processing.py` script automatically:
1. Parses Label Studio JSON consensus annotations
2. Resolves image paths from the annotation data
3. Maps labels (`1_Pronacio`, `2_Neutralis`, `3_Szupinacio`) to class indices (0, 1, 2)
4. Creates stratified train/validation/test splits
5. Saves a manifest CSV file for training

**Label mapping:**
- `1_Pronacio` → Class 0 (Pronation)
- `2_Neutralis` → Class 1 (Neutral)
- `3_Szupinacio` → Class 2 (Supination)

## Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t anklealign:1.0 .
```

### Run

To run the solution, use the following command. You must mount your local data directory to `/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm --gpus all \
    -v /absolute/path/to/your/local/data:/data \
    -v /absolute/path/to/output:/app/output \
    anklealign:1.0 > log/run.log 2>&1
```

**Example (Linux/macOS):**
```bash
docker run --rm --gpus all \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/app/output \
    anklealign:1.0 > log/run.log 2>&1
```

**Example (Windows PowerShell):**
```powershell
docker run --rm --gpus all `
    -v ${PWD}/data:/data `
    -v ${PWD}/output:/app/output `
    anklealign:1.0 > log/run.log 2>&1
```

* Replace `/absolute/path/to/your/local/data` with the actual path to your dataset.
* The `> log/run.log 2>&1` part ensures that all output is saved to `log/run.log`.
* The container is configured to run every step (data preprocessing, training, evaluation, inference).

### Interactive Development with Jupyter Lab

**Linux/macOS:**
```bash
docker run --rm -it --gpus all \
    -v $(pwd)/src:/app \
    -v $(pwd)/notebook:/app/notebook \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/app/output \
    -p 8888:8888 \
    anklealign:1.0 bash
```

Then inside the container:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/notebook
```

## File Structure and Functions

The repository is structured as follows:

```
AnkleAlign/
├── src/                         # Source code
│   ├── run.sh                   # Pipeline script
│   ├── config.py                # Configuration and hyperparameters
│   ├── utils.py                 # Utility functions and logging
│   ├── 01_data_processing.py    # Data loading and preprocessing
│   ├── 02_train.py              # Model training
│   ├── 03_evaluation.py         # Evaluation and metrics
│   └── 04_inference.py          # Inference on new images
├── notebook/                    # Jupyter notebooks for experimentation
├── log/                         # Log files
│   └── run.log                  # Training log output
├── Dockerfile                   # Container environment
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

**Note**: `data/` and `output/` directories are mounted at runtime and not included in the repository.

### Source Files

- **`config.py`**: Configuration file containing all hyperparameters (epochs, batch size, learning rate, etc.) and paths.
- **`utils.py`**: Helper functions including logging setup, model parameter counting, and summary printing.
- **`01_data_processing.py`**: Parses Label Studio JSON annotations, resolves image paths, creates stratified splits.
- **`02_train.py`**: Defines ResNet-18 model, implements training loop with early stopping.
- **`03_evaluation.py`**: Evaluates model on test set, computes metrics, generates confusion matrices.
- **`04_inference.py`**: Runs inference on new images placed in `/data/inference/` directory.

## Pipeline Steps

1. **Data Processing** (`01_data_processing.py`)
   - Parses Label Studio JSON consensus annotations
   - Maps labels to class indices
   - Creates stratified train/val/test splits (70/15/15)
   - Saves dataset manifest to `/app/output/`

2. **Training** (`02_train.py`)
   - Loads pretrained ResNet-18
   - Applies data augmentation (flip, rotation, color jitter)
   - Trains with cross-entropy loss and Adam optimizer
   - Early stopping based on validation accuracy
   - Saves best model to `/app/output/models/`

3. **Evaluation** (`03_evaluation.py`)
   - Loads best model checkpoint
   - Runs inference on test set
   - Computes accuracy, precision, recall, F1-score
   - Generates confusion matrix visualization
   - Saves evaluation report to `/app/output/`

4. **Inference** (`04_inference.py`)
   - Loads trained model
   - Runs predictions on new images in `/data/inference/`
   - Outputs class predictions with confidence scores

## Output Files

After running the pipeline, the following files are generated in the output directory:

| File | Description |
|------|-------------|
| `dataset_manifest.csv` | Image paths, labels, and split assignments |
| `class_info.json` | Class names and label mapping |
| `training_history.csv` | Per-epoch training metrics |
| `training_config.json` | Training hyperparameters |
| `models/best_model.pth` | Best model checkpoint |
| `confusion_matrix.png` | Confusion matrix visualization |
| `confusion_matrix_normalized.png` | Normalized confusion matrix |
| `evaluation_report.json` | Evaluation metrics |
| `test_predictions.csv` | Per-sample predictions |

## Requirements

- Docker
- NVIDIA GPU with CUDA support (for GPU training)
- NVIDIA Container Toolkit (for GPU access in Docker)

## License

This project is for educational purposes as part of a university deep learning course (VITMMA19).
