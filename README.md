# Model Selection on HPC Intro

## background
never used jupyter notebooks or python? - start here https://github.com/matplotlib/AnatomyOfMatplotlib

## setup

Install anaconda python for your platform (if you haven't already) https://www.anaconda.com/download/

from anaconda prompt (shell with all the paths set up)
```
conda create --name modelHPC python=3.6 matplotlib jupyter matplotlib

activate modelHPC

pip install git+https://github.com/AIworx-Labs/chocolate@master

pip install xgboost

conda install sympy

cd ..to desired parent directory for project

git clone https://github.com/strelzoff-erdc/model_selection_on_HPC_intro.git

cd model_selection_on_HPC_intro

jupyter notebook
