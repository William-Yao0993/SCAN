# Stomatal Comprehensive Automated Neural Network(SCAN)
## About
SCAN is an automated tool for measuring stomatal density and aperture size specific for Canola. 

## Installation 
### Prerequisite:
Make sure you have conda and git installed in your local machine. 
1. We use `conda` to manage environment, please download and install Miniconda with correct system version in https://docs.anaconda.com/free/miniconda/

    After installing, initialize your newly-installed Miniconda. 
    
    For Windows: open the "Anaconda Prompt (miniconda3)" program and enter
    ```
    conda init
    ```

    For Mac and Linux: open terminal and enter 
    ```
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
    ```
2. download `git` from this link: https://git-scm.com/downloads 


### Step 1: 
Open a terminal in Mac and Linux or Anaconda Prompt (miniconda3) in Windows, clone the project in your local machine: 
```
git clone https://github.com/William-Yao0993/SCAN.git
cd SCAN
```
### Step 2:
In terminal Configure the working environments:
```
conda env create -f environment.yml
```

## Start SCAN
Make sure the current working directory in terminal is changed inside the SCAN folder, if not: 
```
cd SCAN
```

Activate the working environment: 
```
conda activate SCANenv
```
Run the following command to start SCAN in activated environment:


```
python -m main
```
ps: the first initialization would take 10-20 seconds to start up the application  
### Option 2:
Download the fully packaged executable for usage, but the model running time is slowing down by some reason, and console popups when multiprocessing engaged. 


### Scale bar practical measurement
| Species | Resolution | Magnification | Length1 | Unit1 | Length2 |Unit2 | 
|---------|------------|---------------|---------|-------|---------|------|
|Arabidopsis| 2560 X 1920 | 690.5±1 X | 0.05 | mm | 221 | pixel |
| Canola | 2560 X 1920 | 692±2 X | 0.05 | mm | 222 | pixel |
| Canola | 1280 X 960 | 691±2 X | 0.05 | mm | 112 | pixel |
| Maize | 2560 X 1920 | 413±1 X | 0.1 | mm | 269 | pixel |
| Tobacco | 2560 X 1920 | 690.5±1 X | 0.05 | mm | 221 | pixel |
| Panicum miliacecm | 2560 X 1920 | 413±1 X | 0.1 | mm | 269 | pixel |
| Rice | 2560 X 1920 | 413±2 X | 0.1 | mm | 267 | pixel |
| Rice | 2560 X 1920 | 413±1 X | 0.05 | mm | 227 | pixel |
| Wheat | 2560 X 1920 | 692±1 X | 0.05 | mm | 224 | pixel |
| Wheat | 2560 X 1920 | 412.5±1 X | 0.1 | mm | 267 | pixel |
-------------------------------------------------------------------------------------


## For training 
Please refer to this repository for model weights and training parameters: https://github.com/William-Yao0993/FD_detection

## Citation

If you use this project in your research or wish to refer to the baseline results, please cite us.

```bibtex
@article {Yao2024.06.12.598768,
	author = {Yao, Lingtian and von Caemmerer, Susanne and Danila, Florence R},
	title = {Automated and high throughput measurement of leaf stomatal traits in canola},
	elocation-id = {2024.06.12.598768},
	year = {2024},
	doi = {10.1101/2024.06.12.598768},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/14/2024.06.12.598768},
	eprint = {https://www.biorxiv.org/content/early/2024/06/14/2024.06.12.598768.full.pdf},
	journal = {bioRxiv}
}

```