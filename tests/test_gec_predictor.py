from pathlib import Path
import torch
import requests
from tqdm import tqdm

from allennlp.common.testing import ModelTestCase
from allennlp.predictors import Predictor

# These imports are required so that instantiating the predictor can be done automatically
from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.gec_predictor import GecPredictor


class TestGecPredictor(ModelTestCase):
    """Test class for GecModel"""

    def setUp(self):
        super().setUp()
        # Download weights for model archive
        weights_url = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"
        test_fixtures_dir_path = Path(__file__).parent.parent / "test_fixtures"
        model_path = test_fixtures_dir_path / "roberta_model" / "weights.th"
        if not model_path.exists():
            response = requests.get(weights_url)
            with model_path.open("wb") as out_fp:
                # Write out data with progress bar
                for data in tqdm(response.iter_content()):
                    out_fp.write(data)
        model_path = test_fixtures_dir_path / "roberta_model"
        self.model_path = model_path

    def test_gec_predictor_prediction(self):
        """Test simple prediction integration test with GecPredictor"""

        sentence1 = "I run to a stores every day."
        sentence2 = "the quick brown foxes jumps over a elmo's laziest friend"
        # This micmics how batches of requests are constructed in predict.py's predict_for_file function
        input_data = [sentence1, sentence2]

        gec_model = Predictor.from_path(
            self.model_path, predictor_name="gec-predictor"
        )

        prediction = gec_model.predict_batch(input_data)
        # subject verb agreement is not fixed in the second sentence when predicting using GecModel
        # (i.e.) brown foxes jump
        assert set(prediction[0].keys()) == {
            "logits_labels",
            "logits_d_tags",
            "class_probabilities_labels",
            "class_probabilities_d_tags",
            "max_error_probability",
            "words",
            "labels",
            "d_tags",
            "corrected_words",
        }
        assert prediction[0]["corrected_words"] == [
            "I",
            "run",
            "to",
            "the",
            "stores",
            "every",
            "day.",
        ]
        assert prediction[1]["corrected_words"] == [
            "The",
            "quick",
            "brown",
            "foxes",
            "jumps",
            "over",
            "Elmo's",
            "laziest",
            "friend",
        ]

    def test_gec_predictor_problem_predictions(self):
        """Test problem prediction integration test with GecPredictor"""

        sentence1 = "everyday on tv , there are always the entertainers and athletes dedicating their effort and abilities to entertain audiences ."
        sentence2 = "therefore , in return , these people always hope to earn milionsof dollars every year ."
        sentence3 = "they all have worked hard to present people their best of the best ."
        sentence4 = "therefore , i think some famous althletes and entertainers earn millions of dollars every year is fair , they deserve it ."

        input_data = [sentence1, sentence2, sentence3, sentence4]

        gec_model = Predictor.from_path(
            self.model_path,
            predictor_name="gec-predictor",
        )
        prediction = gec_model.predict_batch(input_data)
        # subject verb agreement is not fixed in the second sentence when predicting using GecModel
        # (i.e.) brown foxes jump
        assert set(prediction[0].keys()) == {
            "logits_labels",
            "logits_d_tags",
            "class_probabilities_labels",
            "class_probabilities_d_tags",
            "max_error_probability",
            "words",
            "labels",
            "d_tags",
            "corrected_words",
        }

        # print(prediction[3]["corrected_words"])

        assert prediction[0]["corrected_words"] == [
            "On",
            "TV",
            ",",
            "there",
            "are",
            "always",
            "entertainers",
            "and",
            "athletes",
            "dedicating",
            "their",
            "effort",
            "and",
            "abilities",
            "to",
            "entertaining",
            "audiences",
            ".",
        ]
        assert prediction[1]["corrected_words"] == [
            "Therefore",
            ",",
            "in",
            "return",
            ",",
            "these",
            "people",
            "always",
            "hope",
            "to",
            "earn",
            "milionsof",
            "dollars",
            "every",
            "year",
            ".",
        ]
        assert prediction[2]["corrected_words"] == [
            "They",
            "all",
            "worked",
            "hard",
            "to",
            "give",
            "people",
            "the",
            "best",
            "of",
            "the",
            "best",
            ".",
        ]
        assert prediction[3]["corrected_words"] == [
            "Therefore",
            ",",
            "I",
            "think",
            "some",
            "famous",
            "althletes",
            "and",
            "entertainers",
            "earning",
            "millions",
            "of",
            "dollars",
            "every",
            "year",
            "is",
            "fair",
            ".",
            "They",
            "deserve",
            "it",
            ".",
        ]
