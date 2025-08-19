# Psychrophilic Microbe Prediction from Species Proteomes

This script predicts whether a species is a psychrophilic (cold-adapted) microbe based on its proteome. The input can be a single FASTA file representing a species proteome or a directory containing multiple species proteomes. The script first generates protein embeddings using the ProtT5 model, then uses a trained PyTorch model for classification, and outputs a CSV file with the prediction results.

## Installation

Create the Conda environment and install dependencies using the provided `environment.yml`:

```bash
# Create environment and install dependencies
conda env create -f environment.yml

# Activate the environment
conda activate CryoClass
```


##  Download ProtT5 Model

This project uses the ProtT5 XL UniRef50 model for protein embeddings. You can download the model using the provided script:

```bash
python utils/download_prot_t5_xl_uniref50.py
```

By default, this script uses the Alibaba Cloud mirror for faster download in China.

You can also download the official ProtT5 XL UniRef50 model from Hugging Face: [ProtT5 XL UniRef50 on Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)



## Usage

Run the prediction script on a species proteome FASTA file or a directory of proteomes:

```bash
cd example/predict
python ../../script/predict.py -f ./test_model -o output.csv -m ../../models/models/best_model_50.pth --sample_n 10
```
### Arguments

| Argument             | Description                                                                                       | Required | Default                         |
| -------------------- | ------------------------------------------------------------------------------------------------- | -------- | ------------------------------- |
| `-f, --fasta`        | Input species proteome FASTA file or directory containing multiple FASTA files                    | Yes      | N/A                             |
| `-o, --output`       | Output CSV file to save prediction results                                                        | Yes      | N/A                             |
| `-m, --model`        | Trained psychrophilic microbe classifier model (PyTorch `.pth` file)                              | Yes      | N/A                             |
| `--protT5_model`     | Path to the ProtT5 XL UniRef50 model directory used for protein embeddings                        | No       | `/models/prot_t5_xl_uniref50` |
| `--sample_n`         | Number of sequences to randomly sample from each FASTA for embedding (default uses all sequences) | No       | `all`                           |
| `--embedding_outdir` | Directory to save protein embeddings as `.pkl` files                                              | No       | `./embeddings`                  |


## Training Psychrophilic Microbe Classifier

This script trains and evaluates models for predicting psychrophilic microbes using protein embeddings. The training configuration is specified in a YAML file.


### Download Train Dataset
You can download the training dataset from Zenodo: [Zenodo Record](https://zenodo.org/records/16899355)


Alternatively, you can use the provided download_dataset.py script in the utils folder to download all files efficiently, with support for multi-threaded downloads and automatic merging of split files:
```bash
python ./utils/download_dataset.py --outdir ./dataset/ --workers 8
```

#### Arguments:

| Argument    | Description                         | Required | Default      |
| ----------- | ----------------------------------- | -------- | ------------ |
| `--outdir`  | Directory to save downloaded files  | No       | `./dataset/` |
| `--workers` | Number of parallel download threads | No       | `8`          |


#### The script will:

- Skip already downloaded files.

- Automatically merge large split files.

- Save all files into the same ./dataset/ folder.



### Usage

Run the training script with the path to a configuration file:

```bash
python ../../script/train.py -c ../../config/config.yaml
```

#### Arguments

| Argument       | Description                                         | Required | Default |
| -------------- | --------------------------------------------------- | -------- | ------- |
| `-c, --config` | Path to the configuration YAML file (`config.yaml`) | Yes      | N/A     |

#### Example

Configuration File (config.yaml)

The configuration file defines the data paths, sampling parameters, model output, and training hyperparameters. Example content:


```bash
# Data paths
file_pkl: "../dataset/embedding_embedding_seq.pkl"
cls_org: "../dataset/cls_org.json"

# Sampling numbers
sample_num_list: [10, 20, 30, 40, 50]

# Model saving & result files
model_output: "./models"
results_file: "model_results.json"
roc_curve_plot: "roc_curve_plot_with_auc.png"

# Training parameters
training:
  num_epochs: 30000
  learning_rate: 1e-6
  early_stopping_patience: 5000
```

- `file_pkl`: Training database containing protein embeddings for all sequences used to train the model.
- `cls_org`: Class labels corresponding to each training sample.
- `sample_num_list`: Number of sequences to sample from each proteome during training.
- `model_output`: Directory to save trained models.
- `results_file`: JSON file to store evaluation results.
- `roc_curve_plot`: File name for the ROC curve plot with AUC.
- `training.num_epochs`: Number of training epochs.
- `training.learning_rate`: Learning rate for the optimizer.
- `training.early_stopping_patience`: Number of epochs to wait for improvement before early stopping.



