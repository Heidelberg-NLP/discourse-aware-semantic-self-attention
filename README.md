# Discourse-Aware Semantic Self Attention for Narrative Reading Comprehension

This repository contains code for the EMNLP-IJCNLP 2019 paper
[Discourse-aware Semantic Self-Attention for Narrative Reading Comprehension](https://arxiv.org/abs/1908.10721).

```
@inproceedings{mihaylov-frank-2019-dassa,
    title = "Discourse-Aware Semantic Self-Attention for Narrative Reading Comprehension",
    author = "Mihaylov, Todor  and Frank, Anette",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1257",
    doi = "10.18653/v1/D19-1257",
    pages = "2541--2552",
}
```

# Setting Up the Environment

1. Create the `dassa` environment using Anaconda

  ```
  conda create -n dassa python=3.6
  ```

2. Activate the environment

  ```
  source activate dassa
  ```

3. Install the requirements in the environment:

```
pip install -r requirements.txt
```

Install pytorch that supports cuda8 cuda 8:
```
pip install torch==0.4.1
```

# Prepare data
Processed narrativeqa data is located in data/ folder. To prepare the data run:
```
bash data/prepare_data.sh
```

# Train models
See [TRAIN.md](TRAIN.md)

# Evaluate trained models
See [EVALUATE.md](EVALUATE.md)