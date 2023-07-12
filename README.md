![ChordGNN_logo](images/chordgnn_logo.png)

# ChordGNN

Official implementation of the paper [Roman Numeral Analysis with Graph Neural Networks: Onset-Wise Predictions from Note-Wise Features](https://arxiv.org/abs/2307.03544), accepted at ISMIR 2023.

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
python ./train/chord_prediction.py  
```

Note use -h to see all the available options.


## Pretrained Models

Pretrained models and logged runs are saved on [wandb](https://wandb.ai/melkisedeath/chord_rec) and can be downloaded from there.

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{karystinaios2023roman,
  title={Roman Numeral Analysis with Graph Neural Networks: Onset-Wise Predictions from Note-Wise Features},
  author={Karystinaios, Emmanouil and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2023}
}
```

### Some of the code is based on the following repositories:

- [AugmentedNet](https://github.com/napulen/AugmentedNet)
- [Voice Separation as Link Prediction](https://github.com/manoskary/vocsep_ijcai2023)


## Aknowledgements

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement
No. 101019375 (“Whither Music?”), and the Federal State of Upper Austria (LIT AI Lab).

