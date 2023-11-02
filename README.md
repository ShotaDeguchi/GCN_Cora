# GCN_Cora

## Results 
Cora classification results with the following:
* Number of GCN layers: 2
* Number of hidden features: 16
* Dropout: 0.5
* Adam optimizer
* Learning rate: 1e-2
* L2 weight decay: 5e-4
* Epochs: 200
* Patience: 10

| loss | accuracy |
| :---: | :---: |
| ![img](results/loss.png) | ![img](results/accuracy.png) |

Test dataset results:
| model | loss | accuracy |
| :---: | :---: | :---: |
| MLP | 1.142 | 61.0 % |
| GCN | 0.961 | 71.5 % |

