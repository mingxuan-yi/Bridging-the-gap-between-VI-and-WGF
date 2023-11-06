
**First, setup the dependencies by**
>pip install -r requirements.txt

## 1.The Illustrative Example
see `notebook/The Illustrative Example.ipynb`.

<img src="https://github.com/YiMX/Bridging-the-gap-between-VI-and-WGF/assets/24216379/b872e5e3-66b0-4b70-88d9-455b0b916a03" width="75%" height="75%">



## 2. The experiment used GMMs as the variational distribution.
run `python banana.py`. Default parameters: number of mixture components `--num_components=5`, particles per component used for Monte Carlo gradients `--num_sample=30`.
<img src="https://github.com/YiMX/Bridging-the-gap-between-VI-and-WGF/assets/24216379/4a9ea9ac-b307-4ad3-a9d0-bb280a0dab13" width="75%" height="75%">


## 3. The experiment on Bayesian logistic regressions.
run `python train_Bayesian_logistic.py --dataset='ionos' --num_epoch=20001`. Change `--dataset` to `'heart', 'pima', 'wine'` for different UCI dataset. 


## 4. To reproduce the experiment in Appendix B.1, run
>python 1d_gmm.py
