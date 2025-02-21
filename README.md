# GrammaFix – AI-Powered Grammar Correction 

This repository provides code for training and testing state-of-the-art models for grammatical error correction with the official PyTorch implementation of the following paper:
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://aclanthology.org/2020.bea-1.16/) <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vitaliy Atrasevych](https://github.com/atrasevych), [Artem Chernodub](https://github.com/achernodub), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> [15th Workshop on Innovative Use of NLP for Building Educational Applications (co-located with ACL 2020)](https://sig-edu.org/bea/2020) <br>

It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Datasets
All the public GEC datasets used in the paper can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Synthetically created datasets can be generated/downloaded [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).<br>
To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
## Pretrained models
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>Confidence bias</th>
    <th>Min error prob</th>
    <th>CoNNL-2014 (test)</th>
    <th>BEA-2019 (test)</th>
  </tr>
  <tr>
    <td>BERT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th">[link]</a></td>
    <td>0.1</td>
    <td>0.41</td>
    <td>61.0</td>
    <td>68.0</td>
  </tr>
  <tr>
    <td>RoBERTa <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>64.0</td>
    <td>71.8</td>
  </tr>
  <tr>
    <td>XLNet <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th">[link]</a></td>
    <td>0.2</td>
    <td>0.5</td>
    <td>63.2</td>
    <td>71.2</td>
  </tr>
</table>

**Note**: The scores in the table are different from the paper's ones, as the later version of transformers is used. To reproduce the results reported in the paper, use [this version](https://github.com/grammarly/gector/tree/fea1532608) of the repository. 

## Train model
To train the model, simply run:
```.bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
                --model_dir MODEL_DIR
```
There are a lot of paramters to specify among them:
- `cold_steps_count` the number of epochs where we train only last linear layer
- `transformer_model {bert,distilbert,gpt2,roberta,transformerxl,xlnet,albert}` model encoder
- `tn_prob` probability of getting sentences with no errors; helps to balance precision/recall
- `pieces_per_token` maximum number of subwords per token; helps not to get CUDA out of memory

In our experiments we had 98/2 train/dev split.



## Model inference
To run your model on the input file use the following command:
```.bash
python predict.py --model_path MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE
```
Among parameters:
- `min_error_probability` - minimum error probability (as in the paper)
- `additional_confidence` - confidence bias (as in the paper)
- `special_tokens_fix` to reproduce some reported results of pretrained models

For evaluation use [M^2Scorer](https://github.com/nusnlp/m2scorer) and [ERRANT](https://github.com/chrisjbryant/errant).

## Text Simplification
This repository also implements the code of the following paper:
> [Text Simplification by Tagging](https://aclanthology.org/2021.bea-1.2/) <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vipul Raheja](https://github.com/vipulraheja), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> [16th Workshop on Innovative Use of NLP for Building Educational Applications (co-located w EACL 2021)](https://sig-edu.org/bea/current) <br>

For data preprocessing, training and testing the same interface as for GEC could be used. For both training and evaluation stages `utils/filter_brackets.py` is used to remove noise. During inference, we use `--normalize` flag.

<table>
  <tr>
    <th></th>
    <th colspan="2">SARI</th>
    <th rowspan="2">FKGL</th>
  </tr>
    <th>Model</th>
    <th>TurkCorpus</th>
    <th>ASSET</th>
  </tr>
  <tr>
    <td>TST-FINAL <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_tst.th">[link]</a></td>
    <td>39.9</td>
    <td>40.3</td>
    <td>7.65</td>
  </tr>
  <tr>
    <td>TST-FINAL + tweaks</td>
    <td>41.0</td>
    <td>42.7</td>
    <td>7.61</td>
  </tr>
</table>

Inference tweaks parameters: <br>
```
iteration_count = 2
additional_keep_confidence = -0.68
additional_del_confidence = -0.84
min_error_probability = 0.04
```
For evaluation use [EASSE](https://github.com/feralvam/easse) package.



## Noticeable works based on GECToR

- Vanilla PyTorch implementation of GECToR with AMP and distributed support by DeepSpeed [[code](https://github.com/cofe-ai/fast-gector)]
- Improving Sequence Tagging approach for Grammatical Error Correction task [[paper](https://s3.eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/sites/8/2021/04/Improving-Sequence-Tagging-Approach-for-Grammatical-Error-Correction-Task-.pdf)][[code](https://github.com/MaksTarnavskyi/gector-large)]
- LM-Critic: Language Models for Unsupervised Grammatical Error Correction [[paper](https://arxiv.org/pdf/2109.06822.pdf)][[code](https://github.com/michiyasunaga/LM-Critic)]

