# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Flax XLM-RoBERTa model. """

from ...file_utils import add_start_docstrings
from ...utils import logging

from ..roberta.modeling_flax_roberta import (
    FlaxRobertaForMaskedLM,
    FlaxRobertaForMultipleChoice,
    FlaxRobertaForQuestionAnswering,
    FlaxRobertaForSequenceClassification,
    FlaxRobertaForTokenClassification,
    FlaxRobertaModel
)

logger = logging.get_logger(__name__)

FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]


XLM_ROBERTA_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen `flax.linen.Module
    <https://flax.readthedocs.io/en/latest/flax.linen.html#module>`__ subclass. Use it as a regular Flax linen Module
    and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__
    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.FlaxPreTrainedModel.from_pretrained` method to load the
            model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaModel(FlaxRobertaModel):
    """
    This class overrides :class:`~transformers.FlaxRobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForMaskedLM(FlaxRobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.FlaxRobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForSequenceClassification(FlaxRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.FlaxRobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForTokenClassification(FlaxRobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.FlaxRobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
""",
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForQuestionAnswering(FlaxRobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.FlaxRobertaForQuestionAnsweringSimple`. Please check the superclass for
    the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForMultipleChoice(FlaxRobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.FlaxRobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
