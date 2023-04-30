# A View From Somewhere: Human-Centric Face Representations

**[Sony AI Inc.](https://ai.sony)**

[Jerone T. A. Andrews](mailto:jerone.andrews@sony.com), Przemyslaw Joniak, 
[Alice Xiang](mailto:alice.xiang@sony.com)

[[`Paper`](https://arxiv.org/pdf/2303.17176.pdf)] [[`Code`](https://github.com/SonyResearch/a_view_from_somewhere)] [[`Dataset`](docs/AVFS_README.md)] [[`BibTeX`](#citing-a-view-from-somewhere)]

Few datasets contain self-identified sensitive attributes, inferring attributes risks
introducing additional biases, and collecting attributes can carry legal risks. Besides,
categorical labels can fail to reflect the continuous nature of human phenotypic
diversity, making it difficult to compare the similarity between same-labeled faces.
To address these issues, we present A View From Somewhere (AVFS)—a dataset of
638,180 human judgments of face similarity. We demonstrate the utility of AVFS
for learning a continuous, low-dimensional embedding space aligned with human
perception. Our embedding space, induced under a novel conditional framework,
not only enables the accurate prediction of face similarity, but also provides a
human-interpretable decomposition of the dimensions used in the human decision-making 
process, and the importance distinct annotators place on each dimension.
We additionally show the practicality of the dimensions for collecting continuous
attributes, performing classification, and comparing dataset attribute disparities.


## Citing A View From Somewhere

If you use A View From Somewhere, please give appropriate credit by using 
the following BibTeX entry:
```
@inproceedings{
andrews2023avfs,
title={A View From Somewhere: Human-Centric Face Representations},
author={Jerone T A Andrews and Przemyslaw Joniak and Alice Xiang},
booktitle={ICLR},
year={2023}
}
```

## Installation
The code was developed using `python=3.10`, `pytorch=1.11` 
and `torchvision=0.12` with CUDA support, as well as `cmake` and `dlib`.

Install A View From Somewhere:

```shell
git clone git@github.com:SonyResearch/A_View_From_Somewhere.git
cd a_view_from_somewhere
conda env create -f avfs.yaml
conda activate avfs
```

## Getting Started

First download A View From Somewhere [model checkpoints](#checkpoints). A model can be 
loaded as follows:

```python
from avfs.build_avfs import load_registered_model
model, annotator_labels = load_registered_model(model_name="avfs_cph")
```
Refer to the [example notebook](notebooks/avfs_model_example.ipynb) for details on 
how use A View From Somewhere models to obtain face embeddings.


## <a name="Dataset"></a>Dataset
Download A View From Somewhere dataset:
```shell
python download.py --avfs_data
```
(The dataset can also be manually downloaded from 
[here](https://zenodo.org/record/7878655).)

A View From Somewhere dataset documentation (i.e., datasheet) can be found 
[here](docs/DATASHEET.pdf) and an overview of the dataset's contents can be found 
[here](docs/AVFS_README.md).


#### Images

A View From Somewhere dataset does not include any images, it instead references to 
images contained in the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).
To obtain the images follow the instructions [here](docs/OAUTH_README.md) to 
configure Google Drive API with OAuth. 

Download FFHQ dataset:
```shell
ulimit -n 10000
python download.py --ffhq_data
```

By downloading the FFHQ dataset you agree that you have read and accepted the 
[FFHQ license agreement](https://github.com/NVlabs/ffhq-dataset/blob/master/LICENSE.txt).


## Models
### Checkpoints
Download A View From Somwhere model checkpoints:
```shell
python download.py --avfs_models
```
(Model checkpoints can also be manually downloaded from 
[here](https://zenodo.org/record/7878655).) 


### Training
Reproduce A View From Somewhere models using a configuration file located 
in `avfs/config`:

```shell
python train.py --cfg_path avfs/config/<filename>.yaml
```

## Privacy
Annotators that contributed to A View From Somewhere may contact 
Sony Europe B.V. at Taurusavenue 16, 2132LS Hoofddorp, Netherlands or 
[privacyoffice.SEU@sony.com](mailto:privacyoffice.SEU@sony.com) to revoke their 
consent in the future or for certain uses.

## License
A View From Somewhere dataset and models are made available under a [Creative Commons 
BY-NC-SA 4.0 license](LICENSE).
