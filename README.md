# Movielens recsys

This dataset describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service.

Folder structure:

```shell
.
├── README.md
├── configs
├── input
├── launchers
├── lightning_logs
├── notebooks
├── output
└── src
    ├── datasets
    ├── lit_models
    ├── models
    ├── plotting
    ├── preparation
```

## Dataset

- Movielens 100k: 100k
- Movielens 25M: 25m

TODO: How has been the data splitted?

# Installation

## Setup environment

Clone repository in a local path and you will have an structure like this

```shell clone-repo
git clone xxxx
cd xxxx
```

Install package dependencies using poetry:

```shell install-packeges-with-poetry
poetry install
```

## Run scripts

To reproduce the experiments use the following scripts

```shell run-script-01
./launchers/01_overfit_batch.sh
```

# Results

## Experiments

This are the results of some of the experiments carried out.

| Model arquitechture           | tain loss 	| vald loss 	| train rmse 	| valid rmse 	|
|------------------------------ |-------------	|-------------- |--------------	|-------------- |
| MatrixFactorization 	        | 24          	| 5,32          | 31,00       	| xx            |
| MatrixFactorizationWithBias 	| 48          	| 5,35          | 31,08       	| xx            |
| NeuralCollaborativeFiltering  | 24          	| 5,59          | 32,48         | xx            |

# Next steps

- Analayze the performance of these models on other datasets, for example in the Movielens 25m
- Convert this problem into a classification problem where classes are 1-5. Would it have better performance?

# Contact ✒️

Carlos Espeleta - @Carlos_Espeleta

LinkedIn: https://www.linkedin.com/in/carlos-espeleta