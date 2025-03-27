# Neural-Variable-Order-FDEs
This repository contains the code for our AAAI 2025 accepted paper, Neural Variable-Order Fractional Differential Equation Networks

## Reproducing Results
To run our code, go to the /src folder.


```bash
python run_GNN_frac_all.py 
--dataset  Cora, Citeseer, Pubmed, CoauthorCS, CoauthorPhy, Computers, Photo
--function laplacian/ transformer
--block constant_frac/ att_frac
--method predictor/ predictor_corrector
--alpha_ode  between (0,1] the value of beta in the paper
--time     integration time
--step_size  

FOR EXAMPLE:

run_GNN_frac_all.py --dataset Cora --function laplacian --block att_frac --cuda 1 --method predictor --epoch 400 --seed 123 --runtime 10 --decay 0.01 --dropout 0.2 --hidden_dim 256 --input_dropout 0.6 --alpha_ode 0.85 --time 40 --step_size 1.0 --lr 0.01

```

## Reference 

Our code is developed based on the following repo:  

The graph neural ODE model is based on the ICLR 2024 [FROND](https://github.com/zknus/ICLR2024-FROND) framework.  


## Citation 

If you find our work useful, please cite us as follows:
```
@INPROCEEDINGS{cui2025neural,
  title={Neural Variable-Order Fractional Differential Equation Networks},
  author={Wenjun Cui and Qiyu Kang and Xuhao Li and Kai Zhao and Wee Peng Tay and Weihua Deng and Yidong Li},
  booktitle={the 39th Annual AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

