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
Arguments

| Argument             | Description                                                                                       | Required | Default                         |
| -------------------- | ------------------------------------------------------------------------------------------------- | -------- | ------------------------------- |
| `-f, --fasta`        | Input species proteome FASTA file or directory containing multiple FASTA files                    | Yes      | N/A                             |
| `-o, --output`       | Output CSV file to save prediction results                                                        | Yes      | N/A                             |
| `-m, --model`        | Trained psychrophilic microbe classifier model (PyTorch `.pth` file)                              | Yes      | N/A                             |
| `--protT5_model`     | Path to the ProtT5 XL UniRef50 model directory used for protein embeddings                        | No       | `../models/prot_t5_xl_uniref50` |
| `--sample_n`         | Number of sequences to randomly sample from each FASTA for embedding (default uses all sequences) | No       | `all`                           |
| `--embedding_outdir` | Directory to save protein embeddings as `.pkl` files                                              | No       | `./embeddings`                  |

