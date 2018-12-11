# Model Selection on HPC Intro

## background
never used jupyter notebooks or python? - start here https://github.com/matplotlib/AnatomyOfMatplotlib

## setup
```conda create --name modelHPC python=3.6 matplotlib jupyter matplotlib

activate modelHPC

pip install git+https://github.com/AIworx-Labs/chocolate@master

pip install xgboost

cd ..to desired parent directory for project

git clone https://github.com/strelzoff-erdc/model_selection_on_HPC_intro.git

cd model_selection_on_HPC_intro

jupyter notebook
