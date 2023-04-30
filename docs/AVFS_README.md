# A View From Somewhere: Human-Centric Face Representations

**[Sony AI Inc.](https://ai.sony)**

[Jerone T. A. Andrews](mailto:jerone.andrews@sony.com), Przemyslaw Joniak, 
[Alice Xiang](mailto:alice.xiang@sony.com)

[[`Paper`](https://arxiv.org/pdf/2303.17176.pdf)] [[`Code`](https://github.com/SonyResearch/a_view_from_somewhere)] [[`Dataset`](AVFS_README.md)] [[`BibTeX`](#citing-a-view-from-somewhere)]

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

## Documentation
A View From Somewhere dataset documentation can be found [here](DATASHEET.pdf).

## Dataset File Structure
A View From Somewhere dataset can also be manually downloaded from 
[here](https://zenodo.org/record/7878655).

The `avfs-dataset-v1.zip` contains an eponymous folder (`avfs-dataset-v1`) with the 
following structure:

    avfs-dataset-v1
    ├── README.html                               # README
    ├── LICENSE.txt                               # License
    ├── DATASHEET.pdf                             # Datasheet
    ├── ooo_train_val.json                        # Train and validation odd-one-out face similarity judgments study data
    ├── ooo_test_same_stimuli.json                # Test odd-one-out face similarity judgments study data (same stimuli)
    ├── ooo_test_novel_stimuli.json               # Test odd-one-out face similarity judgments study data (novel stimuli)
    ├── dimension_topics.json                     # Dimension topics response data
    ├── image_rating.json                         # Image rating response data
    └── prescreener.json                          # Prescreener demographic survey responses from annotators

### JSON Files
The training and validation set of 638,180 quality-controlled 3AFC triplet-judgment 
tuples contains 638,180 unique triplets, i.e., a single judgment per unique triplet. 
The judgments were obtained from 1,645 eligible annotators. 
`ooo_train_val.json` contains the following for each trial of 20 triplets:
```python
{
  "0": {                                                           # Trial index
    "triplet_questions": [[int, int, int], ..., [int, int, int]],  # - 20 triplets of image IDs corresponding to FFHQ images
    "odd_one_out_positions": [int, ..., int],                      # - Odd-one-out position for each triplet
    "split": [int, ..., int],                                      # - Train (1) or validation (0) split
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

The test set partition of 24,060 quality-controlled 3AFC triplet-judgment tuples 
contains 1,000 unique triplets with 22-25 unique judgments per triplet. The judgments 
were obtained from 355 eligible annotators. 
`ooo_test_same_stimuli.json` contains the following for each trial of 20 
triplets:
```python
{
  "0": {                                                           # Trial index
    "triplet_questions": [[int, int, int], ..., [int, int, int]],  # - 20 triplets of image IDs corresponding to FFHQ images
    "odd_one_out_positions": [int, ..., int],                      # - Odd-one-out position for each triplet
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

The test set partition of 80,300 quality-controlled 3AFC triplet-judgment tuples 
contains all possible triplets that could be sampled from 56 images. Each triplet has 
2-3 judgments from 632 eligible annotators. 
`ooo_test_novel_stimuli.json` contains the following for each trial of 20 
triplets:
```python
{
  "0": {                                                           # Trial index
    "triplet_questions": [[int, int, int], ..., [int, int, int]],  # - 20 triplets of image IDs corresponding to FFHQ images
    "odd_one_out_positions": [int, ..., int],                      # - Odd-one-out position for each triplet
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

The 738 quality-controlled human-generated topic labels were obtained for 22 learned 
AVFS-U embedding dimensions, representing a subset from 128 dimensions. The subset of 
dimensions were selected on the basis of their maximal observed value (over the 
stimulus set of 4,921 faces) being sufficiently larger than zero. The topic labels 
were obtained from 84 eligible annotators.
`dimension_topics.json` contains the following for each trial:
```python
{
  "0": {                                                           # Trial index
    "dimension_id": int,                                           # - Index of rated dimension (out of 22 dimensions)
    "original_dimension_id": int,                                  # - Index of rated dimension (out of 128 dimensions),
    "topics": [str, str, str],                                     # - List of three strings based on the labels given by annotator_id
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

The 8,800 quality-controlled human-generated image ratings were obtained for 22 learned 
AVFS-U embedding dimensions, representing a subset from 128 dimensions. The subset of 
dimensions were selected on the basis of their maximal observed value (over the 
stimulus set of 4,921 faces) being sufficiently larger than zero. The ratings were 
obtained from 164 eligible annotators. 
`image_rating.json` contains the following for each trial:
```python
{
  "0": {                                                           # Trial index
    "dimension_id": int,                                           # - Index of rated dimension (out of 22 dimensions)
    "original_dimension_id": int,                                  # - Index of rated dimension (out of 128 dimensions),
    "image_id": int,                                               # - Image ID corresponding to a FFHQ image
    "image_rating": float,                                         # - Image rating (out of 100)
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

`prescreener.json` contains the following for each unique annotator that 
contributed to A View From Somewhere:
```python
{
  "0": {                                                           # Annotator index
    "age": int,                                                    # - Annotator age (in years)
    "ancestry_regions": str,                                       # - Annotator ancestry regions (comma separated)
    "ancestry_subregions": str,                                    # - Annotator ancestry subregions (comma separated)
    "gender_id": str,                                              # - Annotator gender identity
    "nationality": str,                                            # - Annotator nationality
    "annotator_id": str                                            # - Unique annotator ID
  },
  ...
}
```

## Privacy
Annotators that contributed to A View From Somewhere may contact 
Sony Europe B.V. at Taurusavenue 16, 2132LS Hoofddorp, Netherlands or 
[privacyoffice.SEU@sony.com](mailto:privacyoffice.SEU@sony.com) to revoke their 
consent in the future or for certain uses.

## License
A View From Somewhere dataset and models are made available under a [Creative Commons 
BY-NC-SA 4.0 license](../LICENSE).

