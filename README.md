![ChordGNN_logo](images/chordgnn_logo.png)

# ChordGNN

Official implementation of the paper [Roman Numeral Analysis with Graph Neural Networks: Onset-Wise Predictions from Note-Wise Features](), accepted at ISMIR 2023.

This work was conducted at the [Institute of Computation Peception at JKU](https://www.jku.at/en/institute-of-computational-perception/) by [Emmanouil Karystinaios](https://emmanouil-karystinaios.github.io/).

## Abstract


## Installation

Before starting, make sure to have [conda](https://docs.conda.io/en/latest/miniconda.html) installed.

First, create a new environment for ChordGNN:

```shell
conda create -n chordgnn python=3.8
```

Then, activate the environment:

```shell
conda activate chordgnn
```


If you have a GPU available you might want to install the GPU dependencies

```shell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

And configure the system paths with the two following commands (only for Linux, skip on Windows):

```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```



Finally, clone this repository, move to its directory and install the requirements:

```shell
git clone https://github.com/manoskary/chordgnn
cd chordgnn
pip install -r requirements.txt
```

## Analyse Scores

To analyse a score, you need to provide any Score file, such as a MusicXML file:

```shell
python analyse_score.py --score_path <path_to_score>
```

The produced analysis will be saved in the same directory as the score, with the same name and the suffix `-analysis`.
The new score will be in the MusicXML format, and will contain the predicted Roman Numeral Analysis as harmony annoations of a new track with the harmonic rhythm as notes.

## Train ChordGNN

Training a model from scratch generally requires downloading the training data. However, we provide a dataset class that will handle the data downloading and preprocessing.

To train ChordGNN from scratch use:

```shell
python train.py --dataset 
```
