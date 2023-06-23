# robust_MT_evaluation
<<<<<<< HEAD
Repository for ["BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation"](https://arxiv.org/abs/2305.19144), accepted at EAMT 2023.


<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

## TL;DR

This repository is en extension of the original COMET metric, providing different options to enhance it with lexical features. It includes code for **word-level features**, as well as the option to use the same architecture for **sentence-level features**.  
We also provide the data that was used in the experiments and checkpoints for the models presented in the paper: **COMET+aug**, **COMET+SL-feat.** and **COMET+WL-tags**. 
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
To train a COMET model on your data, you can use the following configuration files (these were used for the models presented in the paper):

**COMET** [robust_MT_evaluation/configs/models/regression_metric_original.yaml](../robust_MT_evaluation/configs/models/regression_metric_original.yaml)

**COMET+WL-tags** [robust_MT_evaluation/configs/models/regression_metric_original_with_tags.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_tags.yaml)

**COMET+SL-feat.** [robust_MT_evaluation/configs/models/regression_metric_original_with_feats_bs64.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_feats_bs64.yaml)

**COMET+aug** [robust_MT_evaluation/configs/models/regression_metric_original_with_augmts.yaml](../robust_MT_evaluation/configs/models/regression_metric_original_with_augmts.yaml)


### Languages Covered:

All the above mentioned models are build on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

**Thus, results for language pairs containing uncovered languages are unreliable!**

## COMET Models:

Here are the pretrained models that can be used to evaluate your translations:

- `comet-wl-tags`: **COMET+WL-tags** Regression model with incorporated into the architecture word-level OK / BAD tags that correspond to the subwords of the translation hypothesis.

- `comet-sl-feats`: **COMET+SL-feat.** Regression model that was enhanced with scores obtained from other metrics, BLEU and CHRF, that are used as sentence-level (SL) features in a late fusion manner.

- `comet-aug`: **COMET+aug** Regression model that was trained on a mixture of original and augmented Direct Assessments from WMT17 to WMT20. We use the code provided by the authors of [SMAUG](https://github.com/Unbabel/smaug) and apply their choice of hyperparameters, including the optimal percentage of the augmented data.


**Note:** The range of scores between different models can be totally different. To better understand COMET scores please [take a look at our FAQs](https://unbabel.github.io/COMET/html/faqs.html)

For more information about the available COMET models read our metrics descriptions [here](https://unbabel.github.io/COMET/html/models.html)


## Publications
If you use COMET please cite our work! Also, don't forget to say which model you used to evaluate your systems.

- [Disentangling Uncertainty in Machine Translation Evaluation](https://aclanthology.org/2022.emnlp-main.591.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
=======
Repository for "BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation", accepted at EAMT 2023.

Code will be added soon.
>>>>>>> c46c4a07a098df06cba55aedebc2dd8ac249cf05
