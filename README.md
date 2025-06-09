# "Beyond the past": Leveraging Audio and Human Memory for Sequential Music Recommendation
Source code for our short paper submitted to RecSys 2025.

## Environment

- python 3.9.13
- tensorflow 2.11.0
- tqdm 4.65.0
- numpy 1.24.2
- scipy 1.10.1
- pandas 1.5.3
- toolz 0.12.0

## Dataset
We will release our proprietary data upon acceptance, ensuring anonymity.


## Hyperparameters

Hperparameters for each model are found in the corresponding
configuration file in `configs` directory:
- Number of epochs: 100
- Optimizer: Adam 
- Batch size: 512
- Embedding dimension $d$: 128
- $\alpha = 0.5$ for the Base-Level (BL) module in all ACT-R models. 
- For Transformer-based models (PISA, REACTA): $B=2, H=2$, and $L=30$. 
- Other hyperparameters were tuned via grid search on validation set: 
  - Learning rates: {0.0002, 0.0005, 0.00075, 0.001}
  - $\lambda$: {0.0, 0.3, 0.5, 0.8, 0.9, 1.0} 
  - $\beta$ and $\gamma$: {0.2, 0.4, 0.6, 0.8, 1.0}
