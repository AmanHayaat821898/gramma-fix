"""Basic model. Predicts tags for every token"""
from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import (
    get_text_field_mask,
    sequence_cross_entropy_with_logits,
)
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN

NOOP_INDEX = 0


@Model.register("seq2labels")
class Seq2Labels(Model):
    """
    This ``Seq2Labels`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag (or couple tags) for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    labels_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        predictor_dropout=0.0,
        labels_namespace: str = "labels",
        detect_namespace: str = "d_tags",
        verbose_metrics: bool = False,
        label_smoothing: float = 0.0,
        confidence: float = 0.0,
        del_confidence: float = 0.0,
        min_error_probability: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(Seq2Labels, self).__init__(vocab, regularizer)

        self.label_namespaces = [labels_namespace, detect_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.del_conf = del_confidence
        self.min_error_probability = min_error_probability
        self.incorr_index = self.vocab.get_token_index(
            "INCORRECT", namespace=detect_namespace
        )

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(
            torch.nn.Dropout(predictor_dropout)
        )

        self.tag_labels_projection_layer = TimeDistributed(
            Linear(
                text_field_embedder._token_embedders["bert"].get_output_dim(),
                self.num_labels_classes,
            )
        )

        self.tag_detect_projection_layer = TimeDistributed(
            Linear(
                text_field_embedder._token_embedders["bert"].get_output_dim(),
                self.num_detect_classes,
            )
        )

        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        labels: torch.LongTensor = None,
        d_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        encoded_text = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)
        logits_labels = self.tag_labels_projection_layer(
            self.predictor_dropout(encoded_text)
        )
        logits_d = self.tag_detect_projection_layer(encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes]
        )
        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes]
        )
        error_probs = class_probabilities_d[:, :, self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]

        probability_change = [self.confidence, self.del_conf] + [0] * (
            self.num_labels_classes - 2
        )
        class_probabilities_labels += (
            torch.FloatTensor(probability_change)
            .repeat((batch_size, sequence_length, 1))
            .to(class_probabilities_labels.device)
        )

        output_dict = {
            "logits_labels": logits_labels,
            "logits_d_tags": logits_d,
            "class_probabilities_labels": class_probabilities_labels,
            "class_probabilities_d_tags": class_probabilities_d,
            "max_error_probability": incorr_prob,
        }
        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(
                logits_labels,
                labels,
                mask,
                label_smoothing=self.label_smoothing,
            )
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            for metric in self.metrics.values():
                metric(logits_labels, labels, mask.float())
                metric(logits_d, d_tags, mask.float())
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = []
            for instance in metadata:
                output_dict["words"].append(
                    [word for word in instance["words"] if word != START_TOKEN]
                )
        return output_dict

    @overrides
    def decode(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.

        Parameters
        ----------
        output_dict: Dict[str, torch.Tensor]
            This is expected to have the following keys:
                - logits_labels
                - logits_d_tags
                - class_probabilities_labels
                - class_probabilities_d_tags
                - max_error_probability
                - words

        Returns
        ------
        Dict
            A dictionary containing the following keys:
                - logits_labels
                    Logits for labels indicating the types of corrections
                    to perform.
                - logits_d_tags
                    Logits for labels indicating the presence or absence
                    of grammatical errors.
                - class_probabilities_labels
                    Class probabilities for labels indicating the types
                    of corrections to perform.
                - class_probabilities_d_tags
                    Class probabilities for labels indicating the presence
                    or absence of grammatical errors.
                - max_error_probability
                    A threshold probability that has to be exceeded for an
                    error to be corrected.
                - words
                    The original tokens used to create the instance.
                - labels
                    Labels indicating the types of corrections to perform.
                - d_tags
                    Labels indicating the presence or absence of grammatical errors.
                - corrected_words
                    `words` after applying the correction operations
                    specified in `labels`
        """
        for label_namespace in self.label_namespaces:
            all_predictions = output_dict[
                f"class_probabilities_{label_namespace}"
            ]
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [
                    all_predictions[i] for i in range(all_predictions.shape[0])
                ]
            else:
                predictions_list = [all_predictions]
            all_tags = []

            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [
                    self.vocab.get_token_from_index(
                        x, namespace=label_namespace
                    )
                    for x in argmax_indices
                ]
                all_tags.append(tags)
            output_dict[f"{label_namespace}"] = all_tags

        # This tries to copy what happens in gecmodel _convert and post_process
        batch_size = len(output_dict["labels"])
        output_dict["corrected_words"] = []
        for i in range(batch_size):
            words_in_instance = output_dict["words"][i]
            batch_len = len(words_in_instance)
            # Start of _convert
            probs = output_dict["class_probabilities_labels"][i]
            max_probs = torch.max(probs, dim=-1)
            probs = max_probs[0].tolist()
            indices = max_probs[1].tolist()
            # End of _convert

            # Skip whole sentences if there are no errors
            # No corrections performed
            if max(indices) == 0:
                output_dict["corrected_words"].append(output_dict["words"][i])

            # skip whole sentence if probability of correctness is not high
            elif (
                output_dict["max_error_probability"][i]
                < self.min_error_probability
            ):
                output_dict["corrected_words"].append(output_dict["words"][i])

            else:
                actions_per_token = []
                for j in range(batch_len):
                    if j == 0:
                        token = START_TOKEN
                    else:
                        token = words_in_instance[j]
                    # skip if there is no op performed i.e. no error found
                    if indices[j] == NOOP_INDEX:
                        continue
                    suggested_token_operation = output_dict["labels"][i][j]
                    action = self.get_token_action(
                        index=j,
                        prob=probs[j],
                        sugg_token=suggested_token_operation,
                    )
                    if not action:
                        continue
                    actions_per_token.append(action)
                corrected_sent = get_target_sent_by_edits(
                    output_dict["words"][i], actions_per_token
                )
                output_dict["corrected_words"].append(corrected_sent)
        return output_dict

    def get_token_action(self, index, prob, sugg_token):
        """Get list of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_error_probability or sugg_token in [
            UNK,
            PAD,
            "$KEEP",
        ]:
            return None

        if (
            sugg_token.startswith("$REPLACE_")
            or sugg_token.startswith("$TRANSFORM_")
            or sugg_token == "$DELETE"
        ):
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith(
            "$MERGE_"
        ):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith("$TRANSFORM_") or sugg_token.startswith(
            "$MERGE_"
        ):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index("_") + 1 :]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return metrics_to_return
