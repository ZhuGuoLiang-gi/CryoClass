# Protein Classification from FASTA

This script performs protein sequence classification. The input can be a single FASTA file or a directory containing multiple FASTA files. The script first generates protein embeddings using the ProtT5 model, then uses a trained PyTorch model for prediction, and outputs a CSV file.

## Installation

Create the Conda environment and install dependencies using the provided `environment.yml`:

```bash
# Create environment and install dependencies
conda env create -f environment.yml

# Activate the environment
conda activate CryoClass
```


and any dependencies required for ProtT5 embedding.

Usage
python predict.py -f <FASTA_FILE_OR_DIR> -o <OUTPUT_CSV> -m <MODEL_PTH> --protT5_model <PROTT5_MODEL_DIR> --embedding_outdir <EMBEDDING_DIR> --sample_n <N>

Arguments
Argument	Default	Description
-f, --fasta	Required	Input protein FASTA file or directory
-o, --output	Required	Output CSV file path
-m, --model	../model/best_model_50.pth	Trained PyTorch model file
--protT5_model	../models/prot_t5_xl_uniref50	ProtT5 model directory used for embedding generation
--sample_n	all	Number of sequences randomly sampled from each FASTA file; default is all
--embedding_outdir	./embeddings	Directory to save embeddings (.pkl files)
Examples

Predict a single FASTA file:

python predict.py -f ./test_model/GCA_002968015.faa -o output.csv


Predict all FASTA files in a directory and specify ProtT5 model and embedding output directory:

python predict.py \
    -f ./test_model \
    -o output.csv \
    -m ../model/best_model_50.pth \
    --protT5_model ../models/prot_t5_xl_uniref50 \
    --embedding_outdir ./embeddings \
    --sample_n 50

Notes

ProtT5_embedding.py must exist in the ../utils/ directory, or update the path in embedding_protein accordingly.

The output CSV contains predictions for each sequence, suitable for downstream analysis.

Generating embeddings for many FASTA files may take a long time; using GPU acceleration for ProtT5 is recommended.


If you want, I can also create a **ready-to-run Python script** that generates embeddings and runs predictions in one step, so users don’t need to manually manage `.pkl` files.  
