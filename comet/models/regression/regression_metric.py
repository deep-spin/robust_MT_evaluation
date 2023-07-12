# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
RegressionMetric
========================
    Regression Metric that learns to predict a quality assessment by looking
    at source, translation and reference.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from comet.models.base import CometModel
from comet.models.metrics import RegressionMetrics
from comet.modules import FeedForward, Bottleneck
from transformers.optimization import Adafactor
from comet.models.word_level_utils import convert_word_tags
from comet.models.pooling_utils import average_pooling
from comet.encoders.xlmr import XLMREncoder


class RegressionMetric(CometModel):
    """RegressionMetric:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    :param feature_size: Number of sentence-level features.
    :param word_level_training: True when the word-level tags are passed as input.
    :param model_type: Type of model, 'original'/baseline comet or comet with 'wl-tags'.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4, 
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes_bottleneck: List[int] = [2304, 256],
        hidden_sizes: List[int] = [2304, 768],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        feature_size: Optional[int] = 0,
        word_level_training: Optional[bool] = False,
        word_level_feats: Optional[bool] = False,
        model_type: str = "original", # or "wl-tags"
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "regression_metric",
        )
        self.save_hyperparameters()

        if self.hparams.feature_size > 0:

            if self.hparams.model_type == 'original':
                n = 6

            elif self.hparams.model_type == 'wl-tags':
                n = 8

            self.bottleneck = Bottleneck(
                in_dim = self.encoder.output_units * n, 
                hidden_sizes = [self.hparams.hidden_sizes[0], self.hparams.hidden_sizes_bottleneck[-1]],
                activations = self.hparams.activations,
                dropout = self.hparams.dropout,
            )

            self.estimator = FeedForward(
                in_dim = self.hparams.hidden_sizes_bottleneck[-1] + self.hparams.feature_size,
                hidden_sizes=[self.hparams.hidden_sizes[-1]],
                activations = self.hparams.activations,
                dropout = self.hparams.dropout,
                final_activation = self.hparams.final_activation,
            )
        else:

            if self.hparams.model_type == 'original':
                n = 6

            elif self.hparams.model_type == 'wl-tags':
                n = 8
            
            self.estimator = FeedForward(
                in_dim = self.encoder.output_units * n, 
                hidden_sizes = self.hparams.hidden_sizes,
                activations = self.hparams.activations,
                dropout = self.hparams.dropout,
                final_activation = self.hparams.final_activation,
            )

    def init_metrics(self):
        self.train_metrics = RegressionMetrics(prefix="train")
        self.val_metrics = RegressionMetrics(prefix="val")
    
    def is_referenceless(self) -> bool:
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.hparams.feature_size > 0:
            bott_layers_parameters = [
                {"params": self.bottleneck.parameters() , "lr": self.hparams.learning_rate}
            ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            params = layer_parameters + top_layers_parameters
        if self.hparams.feature_size > 0:
            params += bott_layers_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        # scheduler = self._build_scheduler(optimizer)
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

        if self.hparams.feature_size > 0:
            feats = []
            for feat in sample:
                if feat.startswith("f"):
                    feats.append(sample[feat])
            feature_tensor = torch.as_tensor(feats, dtype=torch.float)
            features = {"custom_features": feature_tensor.T}
        else:
            features = {"custom_features": torch.Tensor()}

        if self.hparams.model_type == 'wl-tags':
            mt_tags_inputs = self.encoder.prepare_sample(sample["mt_tags"])
            mt_tags_inputs = {"mt_tags_" + k: v for k, v in mt_tags_inputs.items()}
            inputs = {**src_inputs, **mt_inputs, **ref_inputs, **mt_tags_inputs, **features}
        else:
            inputs = {**src_inputs, **mt_inputs, **ref_inputs, **features}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets 

    def estimate(
        self,
        src_sentemb: torch.Tensor,
        mt_sentemb: torch.Tensor,
        ref_sentemb: torch.Tensor,
        mt_tags_sentemb: Optional[torch.tensor],
        mt_wordemb: Optional[torch.tensor],
        mt_tags_wordemb: Optional[torch.tensor],
    ) -> Dict[str, torch.Tensor]:

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb
        sum_mt_tags = mt_sentemb + mt_tags_sentemb

        # embedded_sequences = torch.cat(
        #     (mt_sentemb, ref_sentemb, mt_tags_sentemb, prod_mt_tags, diff_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
        #     dim=1,
        # )

        # embedded_sequences = torch.cat(
        #     (mt_sentemb, ref_sentemb, mt_tags_sentemb, prod_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
        #     dim=1,
        # )

        if self.hparams.model_type == 'wl-tags':
            # WL-tags 8, uncomment tokens in xlmr.py
            # version_83
            print('HERE regression_metric.py wl-tags')
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, mt_tags_sentemb, sum_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
                dim=1,
            )

        if self.hparams.model_type == 'original': # suitable for 'comet', 'comet+sl-features' and 'comet+aug'
            # original / original+features 6, comment out tokens in xlmr.py
            print('HERE regression_metric.py original')
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
                dim=1,
            )

        if self.hparams.feature_size > 0:
            bottleneck = self.bottleneck(embedded_sequences)
            seq_feats = torch.cat((bottleneck, custom_features), dim=1)           
            score = self.estimator(seq_feats)
        else:
            score = self.estimator(embedded_sequences)

        # return {"score": self.estimator(embedded_sequences)}
        return score

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        ref_input_ids: torch.tensor,
        ref_attention_mask: torch.tensor,
        custom_features: torch.tensor,
        mt_tags_input_ids: Optional[torch.tensor] = None,
        mt_tags_attention_mask: Optional[torch.tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        # # WL-tags 9, uncomment tokens in xlmr.py (version_84)
        # embedded_sequences = torch.cat(
        #     (mt_sentemb, ref_sentemb, mt_tags_sentemb, prod_mt_tags, diff_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
        #     dim=1,
        # )

        # WL-tags 8, uncomment tokens in xlmr.py (version_82)
        # embedded_sequences = torch.cat(
        #     (mt_sentemb, ref_sentemb, mt_tags_sentemb, prod_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
        #     dim=1,
        # )

        if self.hparams.model_type == 'wl-tags':
            # print('HERE regression_metric.py wl-tags')
            # for the WL model in the paper
            mt_tags_sentemb = self.get_sentence_embedding(mt_tags_input_ids, mt_tags_attention_mask)
            sum_mt_tags = mt_sentemb + mt_tags_sentemb

            # WL-tags 8, uncomment tokens in xlmr.py (version_83)
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, mt_tags_sentemb, sum_mt_tags, prod_ref, diff_ref, prod_src, diff_src),
                dim=1,
            )

        if self.hparams.model_type == 'original': # suitable for 'comet', 'comet+sl-features' and 'comet+aug'
            # original / original+features 6, comment out tokens in xlmr.py
            # print('HERE regression_metric.py original')
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
                dim=1,
            )

        if self.hparams.feature_size > 0:
            bottleneck = self.bottleneck(embedded_sequences)
            seq_feats = torch.cat((bottleneck, custom_features), dim=1)           
            score = self.estimator(seq_feats)
        else:
            score = self.estimator(embedded_sequences)
            # score = self.estimate(src_sentemb, mt_sentemb, ref_sentemb)

        # return score
        return {"score": score}

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        feats = []
        df = pd.read_csv(path)
        flen = self.hparams.feature_size

        if 'mt_tags' in df.columns:
            self.word_level_training = True
            columns = ["src", "mt", "ref", "mt_tags", "score"]
        else:
            columns = ["src", "mt", "ref", "score"]

        for i in range(flen):
            fstring = 'f' + str(i+1)
            print('feature added: ' + str(fstring))
            columns.append(fstring)
            feats.append(fstring)

        df = df[columns]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["mt"] = df["mt"] + ' <myEOS>'

        if 'mt_tags' in df.columns:
            print('Reading and adding MT tags...')
            # df["mt_tags"] = df["mt_tags"].apply(lambda x: x.lower()).astype(str)
            df["mt_tags"] = df["mt_tags"].astype(str)
            df["mt_tags"] = convert_word_tags(df["mt_tags"].to_list())
            # df['mt'] = df['mt'].apply(lambda x: x.strip()) + " </break> " + df["mt_tags"]

        # if self.word_level_feats:
        #     df["mt_tags"] = convert_word_tags_feats(df["mt_tags"].to_list())

        df["score"] = df["score"].astype("float16")
        for feat in feats:
            df[feat] = df[feat].astype(float)

        return df.to_dict("records")
