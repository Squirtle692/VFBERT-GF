# VFBERT-GF

**Leveraging Genomic Bidirectional Encoder Representations from Transformers with a gcForest ensemble framework for Bacterial Virulence Factors Identification**

---

## Data description

Three Virulence Factor datasets are downloaded from [VFDB](http://www.mgc.ac.cn/VFs/), [Victors](https://phidias.us/victors/) and [BV-BRC](https://www.bv-brc.org/), while Non-Virulence Factors are collected from [DEG](http://origin.tubic.org/deg/public/index.php). 


## Pretrain_model

VFBERT, the feature embedding part of VFBERT-GF, is built on [DNABERT](https://github.com/jerryji1993/DNABERT). To run VFBERT-GF, the installation of DNABERT is needed, which could be found at :[https://github.com/jerryji1993/DNABERT]. For model training, NVIDIA GPUs with CUDA are recommended.
The pre-trained weight used in our work is available in [GoogleDrive](https://link-to-google-drive.com).

---

## Instructions

- Collect DNA feature embedding with pre-trained VFBERT.
- Run downstream classifiers to get outputs. All ouputs are concatenated.
- Fed the concatenated output into meta-model gcForest.
