# CryoClass External Validation README

## 1. Purpose
This README describes the external validation design, sample selection, execution workflow, and result interpretation for CryoClass on external samples.

This formal external validation uses two groups of samples:
- `PAc` group: 10 representative strains that are clearly assigned to the cold-adapted lineage
- `NPA` group: 10 representative non-cold-adapted strains selected from different ecological niches

The corresponding result files are:
- `PAc_10_result.csv`
- `NPA_10_result.csv`

## 2. Method Overview
CryoClass is a microbial classification model based on proteomes. The overall workflow is:

1. Input species-level proteome FASTA files
2. Generate protein embeddings using ProtT5
3. Perform pooling across all protein embeddings within the same genome to obtain a genome-level representation
4. Feed the genome-level embedding into the trained neural network model
5. Output the predicted class and its confidence score

In the current binary classification setting of this project:
- `0 = PAc`
- `1 = NPA`

## 3. External Validation Sample Design

### 3.1 PAc Group
The PAc group was selected from the following study:

*Ten novel psychrophilic Flavobacterium species from Tibetan Plateau glaciers define a cryospheric lineage with global cold-origin relatives*  
DOI: `10.1099/ijsem.0.007021`

Select 10 formally included representative PAc strains are listed below:

| Strain | Assembly Accession |
|---|---|
| LB1P62 | GCA_048284325.1 |
| LB2P44 | GCA_048284335.1 |
| LB2P6 | GCA_048284345.1 |
| XS2P39 | GCA_048286515.1 |
| GT2P42 | GCA_048286555.1 |
| LS1P3 | GCA_048286615.1 |
| ZS1P14 | GCA_048286715.1 |
| LB3P6 | GCA_048286755.1 |
| ZB4P13 | GCA_048286775.1 |
| XS1P32 | GCA_048286795.1 |

The corresponding `.faa` files for these 10 PAc strains are stored in the `PAc_10_input` folder.

### 3.2 NPA Group
The NPA group consists of 10 representative non-cold-adapted samples selected for this formal external validation. The selection criteria were:

- To cover different ecological environments, such as host-associated habitats, aquatic environments, food-associated environments, soil, sediment, rhizosphere, and high-temperature environments
- To cover different microbial types, including mesophiles, environmental microbes, pathogen-related microbes, thermophiles, and archaea

The 10 representative NPA strains downloaded from NCBI and included in this study are listed below:

| Genome | Species | Strain | Environment | Topt (°C) |
|---|---|---|---|---|
| GCF_000005845.2 | *Escherichia coli* | K-12 MG1655 | Host-associated | 37 |
| GCF_000006745.1 | *Vibrio cholerae* | N16961 | Aquatic | 37 |
| GCF_000006865.1 | *Lactococcus lactis* | IL1403 | Food | ~30 |
| GCF_000008405.1 | *Thermotoga maritima* | MSB8 | Hydrothermal vent | ~80 |
| GCF_000008565.1 | *Methanocaldococcus jannaschii* | DSM 2661 | Hydrothermal vent | ~85 |
| GCF_000009425.1 | *Geobacillus stearothermophilus* | ATCC 7953 | Thermophilic soil | 60-65 |
| GCF_000013425.1 | *Staphylococcus aureus* | NCTC 8325 | Host-associated | 37 |
| GCF_000092025.1 | *Agrobacterium tumefaciens* | C58 | Rhizosphere | ~28 |
| GCF_000146165.2 | *Shewanella oneidensis* | MR-1 | Sediment | ~30 |
| GCF_000203835.1 | *Streptomyces coelicolor* | A3(2) | Soil | 28-30 |

The corresponding `.faa` files for these 10 NPA strains are stored in the `NPA_10_input` folder.

## 4. Environment Setup and Prediction Command

### 4.1 Environment Installation
Run the following commands in the project root directory `CryoClass-main`:

```bash
conda env create -f environment.yml
conda activate CryoClass
```

If the ProtT5 model has not been downloaded, run:

```bash
python utils/download_prot_t5_xl_uniref50.py
```

### 4.2 External Validation Prediction Command
Run the following command in `CryoClass-main/example/predict`:

```bash
python ../../script/predict.py \
  -f <input_FASTA_directory> \
  -o <output_CSV_file> \
  -m ../../models/models/best_model_50.pth \
  --protT5_model ../../models/prot_t5_xl_uniref50 \
  --sample_n all
```

Parameter description:
- `-f`: input proteome FASTA directory
- `-o`: output result file
- `-m`: trained CryoClass model file
- `--protT5_model`: ProtT5 model directory
- `--sample_n all`: use all protein sequences for embedding to reduce random sampling error

## 5. Output File Description
The prediction output contains at least the following fields:

| Field | Meaning |
|---|---|
| `fasta_name` | Sample name |
| `prediction` | Predicted class |
| `confidence` | Model confidence for the predicted class |

Where:
- `prediction = 0` means predicted as `PAc`
- `prediction = 1` means predicted as `NPA`

## 6. External Validation Results

### 6.1 PAc Group Results
Based on the PAc group output file `PAc_10_result.csv`, the prediction results for the 10 formal PAc samples are as follows:

| fasta_name | prediction | confidence |
|---|---:|---:|
| GCA_048284325.1 | 0 | 0.961095 |
| GCA_048284335.1 | 0 | 0.986339 |
| GCA_048284345.1 | 0 | 0.983643 |
| GCA_048286515.1 | 0 | 0.938863 |
| GCA_048286555.1 | 0 | 0.984646 |
| GCA_048286615.1 | 0 | 0.981209 |
| GCA_048286715.1 | 0 | 0.909939 |
| GCA_048286755.1 | 0 | 0.962509 |
| GCA_048286775.1 | 0 | 0.934673 |
| GCA_048286795.1 | 0 | 0.989703 |

Summary statistics for the PAc group:

| Metric | Value |
|---|---:|
| Number of samples | 10 |
| Predicted as PAc (`prediction=0`) | 10 |
| Accuracy | 100.00% |
| Mean confidence | 0.963262 |
| Minimum confidence | 0.909939 |
| Maximum confidence | 0.989703 |

### 6.2 NPA Group Results
Based on the NPA group output file `NPA_10_result.csv`, the prediction results for the 10 formal NPA samples are as follows:

| fasta_name | prediction | confidence |
|---|---:|---:|
| GCF_000005845.2_ASM584v2_protein | 1 | 0.992361 |
| GCF_000006745.1_ASM674v1_protein | 1 | 0.967013 |
| GCF_000006865.1_ASM686v1_protein | 1 | 0.963584 |
| GCF_000008405.1_ASM840v1_protein | 1 | 0.974679 |
| GCF_000008565.1_ASM856v1_protein | 1 | 0.860626 |
| GCF_000009425.1_ASM942v1_protein | 1 | 0.958801 |
| GCF_000013425.1_ASM1342v1_protein | 1 | 0.925959 |
| GCF_000092025.1_ASM9202v1_protein | 1 | 0.947565 |
| GCF_000146165.2_ASM14616v2_protein | 1 | 0.843566 |
| GCF_000203835.1_ASM20383v1_protein | 1 | 0.902382 |

Summary statistics for the NPA group:

| Metric | Value |
|---|---:|
| Number of samples | 10 |
| Predicted as NPA (`prediction=1`) | 10 |
| Accuracy | 100.00% |
| Mean confidence | 0.933653 |
| Minimum confidence | 0.843566 |
| Maximum confidence | 0.992361 |

### 6.3 Combined Evaluation Metrics
By setting the true label of the PAc group as `0` and the true label of the NPA group as `1`, the overall evaluation metrics of this formal external validation dataset are:

| Metric | Value |
|---|---:|
| Total number of samples | 20 |
| Accuracy | 1.000000 |
| Precision | 1.000000 |
| Recall | 1.000000 |
| Specificity | 1.000000 |
| F1-score | 1.000000 |
| Balanced Accuracy | 1.000000 |
| MCC | 1.000000 |
| ROC AUC | 1.000000 |

The corresponding confusion matrix is:

| True class \\ Predicted class | Predicted PAc (0) | Predicted NPA (1) |
|---|---:|---:|
| True PAc (0) | 10 | 0 |
| True NPA (1) | 0 | 10 |

## 7. Result Interpretation
The results of this external validation indicate that:

1. CryoClass stably recognizes all 10 representative PAc samples, with overall high confidence scores.
2. CryoClass also correctly recognizes all 10 NPA control samples from different ecological niches.
3. In the 20 formally included samples of this validation set, the model completely separates PAc and NPA.
4. Because the NPA group intentionally covers different environments and physiological types, these results suggest that the model has good discriminative ability and a certain level of generalization within the current external sample range.

## 8. Conclusion
Based on the currently included `10 PAc + 10 NPA` external validation samples, CryoClass shows stable performance in this external validation:

- It has stable recognition ability for clearly defined cold-adapted lineage samples
- It has good discriminative ability for typical non-cold-adapted control samples
- The current external validation results support the use of this model for preliminary identification of cold-adapted lineages and screening of external samples
