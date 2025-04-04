# CELLECT
# CELLECT (Contrastive Embedding Learning for Large-scale Efficient Cell Tracking)
The CELLECT tracking framework is designed to extract cell center points from images along with feature vectors representing these points. By incorporating contrastive learning optimization, CELLECT ensures each cell is represented by a distinct feature vector. After training, cell similarity is evaluated based on the distances between their corresponding feature vectors.

The framework comprises three main models:
1. A primary U-Net model, which takes two consecutive frames as input to extract cell segmentations, center points, features, size estimations, and division estimations.
2. An MLP model to evaluate whether multiple cells within the same frame belong to the same cell.
3. A second MLP model to determine whether cells across frames are the same or originate from the same cell, as well as to identify cell division events.



![CELLECT Image](https://github.com/zzz333za/CELLECT-ctc.ver_2024.10/raw/main/CELLECT.png)

### Requirements

To ensure compatibility, please use the following software versions:
- **CUDA Version**: 12.4  
- **Python Version**: 3.11.7
- **torch**==2.3.1



## Running Instructions

### Install required packages by running:

```bash
pip install -r requirements.txt
```


---
### Example
We provide a sample set of commands below to help you quickly set up a clean environment and run the pipeline from scratch:
```bash
cd CELLECT/
conda create -n cellect python=3.11.7 numpy scipy scikit-image -y
conda activate cellect

pip install -r requirements.txt

# Data preprocessing
python con-label-input.py --data_dir ../extradata/mskcc-confocal --out_dir ./ --num 2

# Model training
python s-train.py --data_dir ../extradata/mskcc-confocal --out_dir ./ --train 2 --val 2 --model_dir ./model/

# Model evaluation
python s-test.py --data_dir ../extradata/mskcc-confocal \
  --out_dir ./ \
  --model1_dir ./model/U-ext+-x3rdstr0-149.0-3.4599.pth \
  --model2_dir ./model/EX+-x3rdstr0-149.0-3.4599.pth \
  --model3_dir ./model/EN+-x3rdstr0-149.0-3.4599.pth

```
---


### Data Processing Module

Before training, sparse annotations must be converted into a matrix format suitable for image-based training. To process the data, use the following command:
```bash
python con-label-input.py --data_dir D:/extradata/mskcc-confocal --out_dir C:/Users/try --num 2
```
Parameters:  
- data_dir: Path to the original data folder (source: [Zenodo Dataset](https://zenodo.org/record/6460303))  
- out_dir: Path to save the processed data (requires hundreds of GB)  
- num: Identifier for the subset of the data (the original dataset contains three subsets)
- start: (Optional) Starting frame index (default: 0)
- end: (Optional) Ending frame index (default: 275)
  
#### Data Folder Structure  

```plaintext
extradata/
└── mskcc-confocal/
    ├── mskcc_confocal_s1/
    ├── mskcc_confocal_s2/
    └── mskcc_confocal_s3/

```
---
### Training the Model

To train the model, use the following command with the specified parameters:
```bash
python s-train.py --data_dir D:/extradata/mskcc-confocal   --out_dir C:/Users/try --train 2 --val 2 --model_dir ./model/
```
- data_dir: Path to the folder containing the original data.  
- out_dir: Path to save the processed annotation data.  
- train: Dataset ID (1-3) used for training.  
- val: Dataset ID (1-3) used for validation.  
- model_dir: Path to store the trained model.  

### Test

To run test, use the following command with the specified parameters:
```bash
python s-test.py --data_dir D:/extradata/mskcc-confocal   --out_dir C:/Users/z/Desktop/try --model1_dir ./model/U-ext+-x3rdstr0-149.0-3.4599.pth  --model2_dir ./model/EX+-x3rdstr0-149.0-3.4599.pth --model3_dir ./model/EN+-x3rdstr0-149.0-3.4599.pth
```

Test Parameters    
- data_dir: Path to the test data folder.  
- out_dir: Path to save the output results.    
- model1_dir, model2_dir, model3_dir: Paths to the three trained model weight files.
- cpu: Use CPU for inference only. Automatically enabled if no compatible GPU is available.



### Additional Information

In addition, we developed an advanced version of CELLECT optimized for the Cell Tracking Challenge (CTC). This version demonstrates better performance in tracking and segmentation compared to the original, with enhanced precision in the instance segmentation module tailored for high-density embryonic cells (the original method was based on sparse annotations, focusing on center positions and segmentation areas without specifically separating and storing individual cells). As a result, this version is significantly slower, but the performance can be improved by adjusting the file type and precision of the output segmentation files.

The advanced version is available at [CELLECT-ctc.ver](https://github.com/zzz333za/CELLECT-ctc.ver)
