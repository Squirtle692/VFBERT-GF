# VFBERT-GF

**Leveraging Genomic Bidirectional Encoder Representations from Transformers with a gcForest ensemble framework for Bacterial Virulence Factors Identification**

---

## Data description

Three Virulence Factor datasets are downloaded from [VFDB](http://www.mgc.ac.cn/VFs/), [Victors](https://phidias.us/victors/) and [BV-BRC](https://www.bv-brc.org/), while Non-Virulence Factors are collected from [DEG](http://origin.tubic.org/deg/public/index.php). Additionally, ncDNA data is sourced from [RNAcentral](https://rnacentral.org/).


## Pretrain_model

VFBERT, the feature embedding part of VFBERT-GF, is built on [DNABERT](https://github.com/jerryji1993/DNABERT). To run VFBERT-GF, the installation of DNABERT is needed, which could be found at :[https://github.com/jerryji1993/DNABERT]. For model training, NVIDIA GPUs with CUDA are recommended.
The pre-trained weight used in our work is available in [GoogleDrive](https://drive.google.com/file/d/1JY2GhjqQIx23JNivn3yiSP7XAgBjQKyV/view?usp=sharing).

---

## Instructions

- Collect DNA feature embedding with pre-trained VFBERT.
- Run downstream classifiers to get outputs. 
- All ouputs are concatenated and fed into meta-model gcForest.

## Data

- Data used to compare different embedding methods - [Train](https://drive.google.com/drive/folders/1QyIiCnCNUSbXTpbPbT9xaW1hWb72JsLD?usp=sharing) and [Test](https://drive.google.com/drive/folders/1RPz6BIoQIHyI2XhXdKTHniecuU5CYTFf?usp=sharing)
- [Train](https://drive.google.com/drive/folders/14GCsk3wQLw-w2WcLAa5zCaAt206Q2hgd?usp=sharing) and [Test](https://drive.google.com/drive/folders/13q4W6L0gd_ThWj1XnbwZ2-DYy1KnmklH?usp=sharing) with different similarity thresholds
- Data uesed to compare VFBERT-GF and other SOTA models - [Train](https://drive.google.com/drive/folders/1u8B25J5BIUR1FngwubBp4g859IIFyasn?usp=sharing) and [Test](https://drive.google.com/drive/folders/1EHRh1EiKOOftt_VIw5sY_EVIY9fXlmA2?usp=sharing)
