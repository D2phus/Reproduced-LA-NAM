# Reproduced LA-NAM
| **[Overview](#overview)**
| **[Reference](#Reference)**

## Setup
The `nam`, `laplace-torch`, and `backpack` packages are required. 
```
pip install nam
pip install laplace-torch
pip install backpack-for-pytorch
```
## Overview
The main project structure is shown below. 
    
```
Reproduced-LA-NAM/
    LANAM/
        models/
            lanam.py
            bayesian_linear_regression.py
            utils.py
            featurenn.py
            nam/
                featurenn.py
                nam.py
            activation/
                exu.py
                relu.py
                gelu.py
        trainer/
            wandb_train.py
            marglik_training.py
            mle_training.py
            feature_interaction_selection.py
            test.py
            nam_trainer/
                metrics.py
                train.py
                epoch.py
                losses.py
        data/
            dataset.py
            generator.py
            utils.py
            base.py
            datasets/
        utils/
            wandb.py
            plotting.py
            output_filter.py
        config/
            base.py
            default.py
        extension/
            backpack/
                custom_modules/
                derivatives/
                firstorder/
    experiments/
        lanam_sweep.py
        lanam_sweep.sh
        nam_sweep.py
        nam_sweep.sh
    example.ipynb
    
        
```
1. `LANAM.models` provides implementation of models, including LA-NAM (`LANAM.models.lanam`), vanilla NAM (`LANAM.models.nam.nam`), and different hidden units (`exu`, `gelu` and `relu`). 
2. training and test codes are included in trainer `LANAM.trainer`: 
    - different learning algorithms for LA-NAM: MLE training (`LANAM.trainer.mle_training`), joint optimization of marginal likelihood and prediction (`LANAM.trainer.marglik_training`), and feature interaction selection (TODO). Additionally, W&B management codes for joint optimization is provided (`LANAM.trainer.wandb_train`).
    - training algorithm with W&B management for the vanilla NAM (`LANAM.trainer.nam_trainer.train`).
    - a standard test for models `LANAM.trainer.test`, where average MSE loss is used for regression, and average Accuracy is used for classification.
3. Data should be preprocessed as `LANAM.data.base.LANAMDataset` or `LANAM.data.base.LANAMSyntheticDataset` instance. `LANAM.data.base.dataset` provides some loader functions for different raw data, including `load_synthetic_data()` for general synthetic examples and `load_concurvity_data()` for concurvity examples.
4. Plotting function and W&B logger and loader can be found under `LANAM.utils`. Besides, a single-output customized model built by removing selected subnets from the trained LA-NAM is presented in `LANAM.utils.output_filter`, which will be used in feature interaction selection process.
5. `LANAM.config` comprises an experiment configuration class (`LANAM.config.base`) and the default configuration for LA-NAM and vanilla NAM (`LANAM.config.default`). 
6. since the backpropagation computing for some modules (e.g. hidden unit `ExU`) is not directly supported by the `BackPACK` package, `LANAM.extensions.backpack` implements some extensions and custom modules based on `BackPACK`, including partial derivatives calculation (`LANAM.extensions.backpack.derivatives`), first-order extensions (`LANAM.extensions.backpack.firstorder`), and the custom modules (`LANAM.extensions.backpack.custom_modules`).
6. Under `experiments` folder, codes to run hyper-parameter searching with W&B for LA-NAM and vanilla NAM can be found (`experiments/lanam_sweep.py` and `experiments/nam_sweep.py`), as well as batch scripts for parallelism (e.g. individual job for each activation function).
7. Finally, an regression example on toy dataset is presented in `example.ipynb`

## Reference
The Laplace-approximated NAM (LA-NAM):
```bibtex
@misc{bouchiat2023laplaceapproximated,
      title={Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference}, 
      author={Kouroche Bouchiat and Alexander Immer and Hugo Yèche and Gunnar Rätsch and Vincent Fortuin},
      year={2023},
      eprint={2305.16905},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
The PyTorch laplace package:
```bibtex
@inproceedings{laplace2021,
  title={Laplace Redux--Effortless {B}ayesian Deep Learning},
  author={Erik Daxberger and Agustinus Kristiadi and Alexander Immer 
          and Runa Eschenhagen and Matthias Bauer and Philipp Hennig},
  booktitle={{N}eur{IPS}},
  year={2021}
}
```
The PyTorch backpropagation package:
```bibtex
@inproceedings{dangel2020backpack,
    title     = {Back{PACK}: Packing more into Backprop},
    author    = {Felix Dangel and Frederik Kunstner and Philipp Hennig},
    booktitle = {International Conference on Learning Representations},
    year      = {2020},
    url       = {https://openreview.net/forum?id=BJlrF24twB}
}
```
