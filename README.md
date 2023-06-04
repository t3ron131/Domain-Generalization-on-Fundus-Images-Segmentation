# Domain Generalization on Fundus Images Segmentation
***Colaboration with [UrMBCMRabbont](https://github.com/UrMBCMRabbont)***

*The Final Project 2 of HKUST ELEC4010N - Artificial Intelligence for Medical Image Analysis*

Implementing domain generalization of multi-class segmentation on fundus images segmentation dataset by Fourier Augmented Co-Teacher (FACT) model and U-Net.

For more high-level details, read the Project 2 part of the [presentation slides](./Presentation.pdf) and the [report](./Report.pdf).

There are many different combinations of results, but in general, the mean Dice, OC test ASD and OD test ASD are improved.

## Prerequisites
Download and unzip data from the link [Fundus dataset](https://drive.google.com/u/0/uc?id=1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH&export=download).

Place the `Fundus` folder (not `Fundus-doFE`)  into the main directory. If running in Colab, change the paths accordingly and run the following commands in the notebook:

```python
from google.colab import drive
drive.mount('/content/gdrive')
%cd "/content/gdrive/MyDrive/Colab Notebooks/.../your_project_folder"
!unzip "/content/gdrive/MyDrive/Colab Notebooks/.../your_project_folder/Fundus-doFE.zip" -d "/content/"
```

Install the additional libraries by:

```python
!pip install segmentation-models-pytorch
!git clone https://github.com/deepmind/surface-distance.git
!pip install surface-distance/
```

For the requirements of `segmentation-models-pytorch`, install the packages by `pip install -r requirements.txt` or in the notebook:

```python
!pip install torchvision>=0.5.0
!pip install pretrainedmodels==0.7.4
!pip install efficientnet-pytorch==0.7.1
!pip install timm==0.6.13
!pip install tqdm
!pip install pillow
```

These parts are included in the first two code cells in the notebook.

## Notebook Outline
0. For Colab
1. Import
2. Fundus Dataset
3. Segmentation Baseline
    1. U-Net
    2. Average Surface Distance (ASD)
4 Baseline Experiment
    1. Training
    2. Results
    3. Evaluation
5. FACT
    1. Utilities
    2. Fourier Augmentation
    3. Mean Teacher Model
    4. Training
    5. Results
    6. Evaluation

## Reference
Wang, S., Yu, L., Li, K., Yang, X., Fu, C.-W., Heng, P.-A. (2020). DoFE: Domain-oriented Feature Embedding for
Generalizable Fundus Image Segmentation on Unseen Datasets. IEEE Transactions on Medical Imaging.
(https://github.com/emma-sjwang/Dofe)

Xu, Q., Zhang, R., Zhang, Y., Wang, Y., Tian, Q. (2021). A Fourier-Based Framework for Domain Generalization. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
(https://github.com/MediaBrain-SJTU/FACT)

Laine, S., & Aila, T. (2017). Temporal Ensembling for Semi-Supervised Learning. International Conference on
Learning Representations (ICLR). arXiv:1610.02242
