# BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation
Repository for ["BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation"](https://arxiv.org/abs/2305.19144), accepted at EAMT 2023.

## TL;DR

This repository is en extension of the original COMET metric, providing different options to enhance it with lexical features. 

It includes code for **word-level** and **sentence-level features**.  We also provide the data that was used in the experiments and checkpoints for the models presented in the paper: **COMET+aug**, **COMET+SL-feat.** and **COMET+WL-tags**. 

We used COMET v1.0 as the basis for this extension.

Soon: we will add similar checkpoints but for a newer COMET v2.0.

## Quick Installation

COMET requires python 3.8 or above. In our experiments we are using python 3.8.

Detailed usage examples and instructions for the COMET metric can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

To develop locally install [Poetry](https://python-poetry.org/docs/#installation) (`pip install poetry`) and run the following commands:
```bash
git clone https://github.com/deep-spin/robust_MT_evaluation.git
cd robust_MT_evaluation
poetry install
```

## Important commands

### Training your own Metric:

- To train a new model use:

    ```bash
    comet-train --cfg configs/models/{your_model_config}.yaml
    ```

### Scoring MT outputs:

- To score with your trained metric use:

    ```bash
    comet-score --model <path_to_trained_model> -s src.txt -t mt.txt -r ref.txt --to_json <path_where_to_save_the_scores>
    ```

- If you used word-level tags during training, then add ```-wlt <path_to_wlt_for_mt>```

    ```bash
    comet-score --model <path_to_trained_model> -s src.txt -t mt.txt -r ref.txt -wlt <path_to_wlt_for_mt> --to_json <path_where_to_save_the_scores>
    ```

- If you used sentence-level features during training, then add ```-f <path_to_features_for_mt>```

    ```bash
    comet-score --model <path_to_trained_model> -s src.txt -t mt.txt -r ref.txt -f <path_to_features_for_mt> --to_json <path_where_to_save_the_scores>
    ```

**Note:** Please contact ricardo.rei@unbabel.com if you wish to host your own metric within COMET available metrics!

### COMET configurations
To train a COMET model on your data you can use the following configuration files:

**COMET**
[robust_MT_evaluation/configs/models/regression_metric_original.yaml](../robust_MT_evaluation/configs/models/regression_metric_original.yaml)

**COMET+WL-tags**
[robust_MT_evaluation/configs/models/regression_metric_original_with_tags.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_tags.yaml)

**COMET+SL-feat.**
[robust_MT_evaluation/configs/models/regression_metric_original_with_feats_bs64.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_feats_bs64.yaml)

**COMET+aug**
[robust_MT_evaluation/configs/models/regression_metric_original_with_augmts.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_augmts.yaml)


## COMET Models

Here are the pretrained models that can be used to evaluate your translations:

- [`comet-wl-tags`](https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt23/comet-wl-tags.tar): Regression model with incorporated into the architecture word-level OK / BAD tags that correspond to the subwords of the translation hypothesis. (**COMET+WL-tags**)

- [`comet-sl-feats`](https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt23/comet-sl-feats.tar): Regression model that was enhanced with scores obtained from other metrics, BLEU and CHRF, that are used as sentence-level (SL) features in a late fusion manner. (**COMET+SL-feat.**)

- [`comet-aug`](https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt23/comet-aug.tar): Regression model that was trained on a mixture of original and augmented Direct Assessments from WMT17 to WMT20. We use the code provided by the authors of [SMAUG](https://github.com/Unbabel/smaug) and apply their choice of hyperparameters, including the optimal percentage of the augmented data. (**COMET+aug**)


**Note:** The range of scores between different models can be totally different. To better understand COMET scores please [take a look at these FAQs](https://unbabel.github.io/COMET/html/faqs.html)

**Note #2:** The word-level tags can be generated in different ways. To generate tags for subwords instead of tokens we use a modified version of [WMT word-level quality estimation task](https://github.com/deep-spin/qe-corpus-builder).

## Related Publications

- [Disentangling Uncertainty in Machine Translation Evaluation](https://aclanthology.org/2022.emnlp-main.591.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)

## Citation

If you found our work/code useful, please consider citing our paper:

```bibtex
@article{glushkova2023bleu,
  title={BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation},
  author={Glushkova, Taisiya and Zerva, Chrysoula and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2305.19144},
  year={2023}
}
```

## Acknowledgments

This code is largely based on the [COMET](https://github.com/Unbabel/COMET) repo by Ricardo Rei.
