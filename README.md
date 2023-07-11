![ChordGNN_logo](images/chordgnn_logo.png)

# ChordGNN

Official implementation of the paper [Roman Numeral Analysis with Graph Neural Networks: Onset-Wise Predictions from Note-Wise Features](https://arxiv.org/), accepted at ISMIR 2023.

This work was conducted at the [Institute of Computation Peception at JKU](https://www.jku.at/en/institute-of-computational-perception/) by [Emmanouil Karystinaios](https://emmanouil-karystinaios.github.io/).

## Abstract
Roman Numeral analysis is the important task of identifying chords and their functional context in pieces of tonal music. 
This paper presents a new approach to automatic Roman Numeral analysis in symbolic music. While existing techniques rely on an intermediate lossy representation of the score, we propose a new method based on Graph Neural Networks (GNNs) that enable the direct description and processing of each individual note in the score. 
The proposed architecture can leverage notewise features and interdependencies between notes but yield onset-wise representation by virtue of our novel edge contraction algorithm. 
Our results demonstrate that _ChordGNN_ outperforms existing state-of-the-art models, achieving higher accuracy in Roman Numeral analysis on the reference datasets. 
In addition, we investigate variants of our model using proposed techniques such as NADE, and post-processing of the chord predictions. The full source code for this work is available at [https://github.com/manoskary/chordgnn](https://github.com/manoskary/chordgnn)

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


If you have a GPU available you might want to install the GPU dependencies follow [this link](https://pytorch.org/) to install the appropriate version of Pytorch:
In general for CPU version:
```shell
conda install pytorch==1.12.0 cpuonly -c pytorch
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
