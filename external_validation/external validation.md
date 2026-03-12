# CryoClass External Validation Workflow 

---

# 1. Project Overview

CryoClass is a deep learning-based microbial classification tool used to identify whether a microorganism belongs to a **cryophilic (low-temperature adapted) lineage**. The tool extracts genome-level features through protein sequence language-model embeddings and performs classification using a neural network model.

The overall computational workflow is:

Genome protein sequences (`.faa`)  
-> Protein language model embedding generation (ProtT5)  
-> Genome-level embedding aggregation (pooling)  
-> Deep neural network classification  
-> Output of class prediction and confidence score

In this workflow, each protein sequence is first transformed into a high-dimensional vector representation. These vectors jointly capture structural, evolutionary, and functional characteristics of proteins. Then multiple protein embeddings are aggregated by pooling into a genome-level representation, which is finally fed into the trained classifier for prediction.

---

# 2. Objective of External Validation

This study uses the CryoClass model to perform external validation on a set of **newly discovered psychrophilic microorganisms**, in order to evaluate the model's ability to recognize microbes from extremely low-temperature environments.

Validation data source: https://doi.org/10.1099/ijsem.0.007021

**Ten novel psychrophilic Flavobacterium species from Tibetan Plateau glaciers define a cryospheric lineage with global cold-origin relatives**

Ten representative newly described psychrophilic strains reported in this study and isolated from Tibetan Plateau glaciers were selected as the external validation dataset:

| Organism Qualifier | Assembly Accession |
|---|---|
| strain: LB1P62 | GCA_048284325.1 |
| strain: LB2P44 | GCA_048284335.1 |
| strain: LB2P6 | GCA_048284345.1 |
| strain: XS2P39 | GCA_048286515.1 |
| strain: GT2P42 | GCA_048286555.1 |
| strain: LS1P3 | GCA_048286615.1 |
| strain: ZS1P14 | GCA_048286715.1 |
| strain: LB3P6 | GCA_048286755.1 |
| strain: ZB4P13 | GCA_048286775.1 |
| strain: XS1P32 | GCA_048286795.1 |

These strains are considered members of the **PAc (Polar-Alpine cryophilic lineage)**.

The goal is to use CryoClass to predict these newly discovered strains and evaluate whether the model can correctly identify their low-temperature adaptation characteristics.

---

# 3. Software Environment Installation

CryoClass is installed using a Conda environment.

Create environment:

```bash
conda env create -f environment.yml
```

Activate environment:

```bash
conda activate CryoClass
```

---

# 4. Input Data Preparation

CryoClass requires **protein FASTA files (`.faa`)** as input, with one protein FASTA file per genome.

Our input directory:

```text
input_10/
GCA_048284325.1.faa
GCA_048284335.1.faa
GCA_048284345.1.faa
GCA_048286515.1.faa
GCA_048286555.1.faa
GCA_048286615.1.faa
GCA_048286715.1.faa
GCA_048286755.1.faa
GCA_048286775.1.faa
GCA_048286795.1.faa
```

These protein sequences usually come from genome annotation data provided by the NCBI genome database.

---

# 5. Prediction Workflow

Enter prediction directory:

`CryoClass-main/example/predict`

Run prediction command:

```bash
python ../../script/predict.py \
-f ./input_10 \
-o output.csv \
-m ../../models/models/best_model_50.pth \
--protT5_model ../../models/prot_t5_xl_uniref50 \
--sample_n all
```

---

# 6. Parameter Description

| Parameter | Description |
|---|---|
| `-f` | Input protein FASTA directory |
| `-o` | Output result CSV file |
| `-m` | Trained CryoClass model file |
| `--protT5_model` | ProtT5 embedding model path |
| `--sample_n` | Number of proteins used for embedding per genome |

---

# 7. Output Result Description

Prediction results are saved in:`output.csv`

Field description:

| Field | Meaning |
|---|---|
| `fasta_name` | Input genome name |
| `prediction` | Classification result |
| `confidence` | Prediction probability |

Class label meaning:

`0 = PAc`  
`1 = NPA`

`confidence` indicates the model confidence for the predicted class.

Our output of the external validation dataset :

```text
fasta_name	prediction	confidence
GCA_048284325.1	0	0.961095
GCA_048284335.1	0	0.986339
GCA_048284345.1	0	0.983643
GCA_048286515.1	0	0.938863
GCA_048286555.1	0	0.984646
GCA_048286615.1	0	0.981209
GCA_048286715.1	0	0.909939
GCA_048286755.1	0	0.962509
GCA_048286775.1	0	0.934673
GCA_048286795.1	0	0.989703
```

Current prediction results show:

`prediction = 0`  
`confidence ~= 0.9-0.99`

This indicates that CryoClass predicts these genomes as **PAc**, which is consistent with published findings.

---

# 8. Protein Embedding Generation Process

During prediction, CryoClass uses the ProtT5 model to generate an embedding vector for each protein sequence.

Embedding files are usually cached in `.pkl` format.

Example:

```text
embeddings/
GCA_048286715.1_0001.pkl
GCA_048286715.1_0002.pkl
...
```

Each embedding file corresponds to the vector representation of one protein sequence.

The system then aggregates these embeddings to obtain a genome-level embedding for downstream classification.

---

# 9. Detailed Explanation of `sample_n`

To improve result reliability, the recommended setting is `--sample_n all`, which uses **all protein sequences** in a genome. This reduces sampling error and improves prediction stability.

Computation flow:

Genome proteins  
-> All protein embeddings  
-> Embedding pooling  
-> Classification prediction

---

# 10. Summary

This study performs external validation of CryoClass on psychrophilic microorganisms from Tibetan Plateau glaciers.

Main workflow:

Protein FASTA input  
-> ProtT5 embedding computation  
-> Genome-level embedding aggregation  
-> Deep learning classification prediction

The `sample_n` parameter has a significant impact on prediction results.To obtain more reliable genome-level representations, it is recommended to use `sample_n = all`.
