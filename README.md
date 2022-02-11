# PreSumm [Edited version]

- This code is for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)
- The original codes are available in the [PreSumm repository by nlpyang](https://github.com/nlpyang/PreSumm)

This repository is going to provide a full explanation of how to train a PreSumm model. There are many changes in comparisons of the original code, and the changes are explained in the [CHANGES.md](https://github.com/parsa-abbasi/PreSumm/blob/master/CHANGES.md) file.

## Installation

### Clone

First of all, you should clone this repository using the following command:

```sh
git clone https://github.com/parsa-abbasi/PreSumm.git
```

```sh
cd PreSumm
```

### Requirements

The PreSumm requires the following libraries:

- torch==1.1.0
- pytorch_transformers (an old version of the transformers library by [HuggingFace](https://huggingface.co))
- tensorboardX
- multiprocess
- pyrouge

All of them can be installed using the following command:
```sh
pip uninstall torchvision
```
```sh
pip install -r requirements.txt
```

The original code was based on the *pytorch_transformers* which is the old version of *HuggingFace's transformers*. However, we should replace that with the *transformer* library, as we want to use state-of-the-art transformer models. On the other hand, some functions are deprecated or renamed in the latest version of this library. The best version I found suitable for this project is v2.1.0. You can install it using the following code:

```sh
pip install transformers==2.1.0
```

You can test the following python codes, to make sure that the transformers library is correctly installed:

```python
import transformers
transformers.__version__
from transformers import AutoModel
```

However, the default installation of the pyrouge library can be cause some errors. Therefore you should install pyrouge with a few tricky  commands (based on this [colab notebook](https://colab.research.google.com/drive/1-vAnr3d3W8GtqSCn4MwjrdQrzN0uCXzx?usp=sharing#scrollTo=flpYGUs0cNZh)):

```sh
pip install pyrouge --upgrade
pip install https://github.com/bheinzerling/pyrouge/archive/master.zip
pip install pyrouge
pip show pyrouge
git clone https://github.com/andersjo/pyrouge.git
from pyrouge import Rouge155
pyrouge_set_rouge_path 'pyrouge/tools/ROUGE-1.5.5'
```

 ```sh
 sudo apt-get install libxml-parser-perl
 ```

```sh
cd pyrouge/tools/ROUGE-1.5.5/data
```

```sh
rm WordNet-2.0.exc.db # only if exist
```

```sh
cd WordNet-2.0-Exceptions
```

```sh
rm WordNet-2.0.exc.db # only if exist
```

```sh
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
```

```sh
cd ../
```

```sh
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

Make sure that Java is installed on your device, then run the following commands to download and install the Stanford CoreNLP that will be used in the preprocessing steps:

```sh
cd stanford
wget https://nlp.stanford.edu/software/stanford-corenlp-4.2.1.zip
unzip stanford-corenlp-4.2.1.zip
```

Also, run the following command to set the classpath of this java application. You should change the `ABSOLOUTEPATH` with the path of the `stanford-corenlp-4.2.1.jar` file (for example something like this: `/tf/data/PreSumm/stanford/stanford-corenlp-4.2.1/stanford-corenlp-4.2.1.jar`)

```sh
export CLASSPATH=ABSOLOUTEPATH
```

Run the following sample code to make sure that the CoreNLP works fine:

```sh
echo "Tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
```

## Bert Model

Now you should download a preferred Bert model and put its files in the `bert_model` folder. There are many models for different languages in the *HuggingFace* library. Feel free to download the one that is suitable for your task.

The default Bert model in the original PreSumm was bert_base_uncased which is accessible on [this page](https://huggingface.co/bert-base-uncased). You need to download only the files with `*.json` and `*.txt` format, also the `pytorch_model.bin` file.

### ParsBert

For example, you just need to download the `pytorch_model.bin`, `config.json`, and the `vocab.txt` of this model if you want to use the *ParsBert* model. The download link of these files is accessible on the [*HuggingFace* page](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased/tree/main) of the model.

```sh
cd bert_model
```

```sh
wget https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased/resolve/main/config.json
wget https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased/resolve/main/vocab.txt
```

<font color='red'>**Note:**</font> The following tokens must be present in the `vocab.txt` file. Otherwise, you need to replace some of the words with the following tokens.

```markdown
[unused0]
[unused1]
[unused2]
[unused3]
[unused4]
[unused5]
[unused6]
```

## Dataset

### Input Data

The `raw_data` folder is where you should put your data sets. The structure of this folder is like the following:

* raw_data
  * test
    * stories
      * test_0.story
      * test_1.story
      * ...
  * train
    * stories
      * train_0.story
      * train_1.story
      * ...
  * val
    * stories
      * val_0.story
      * val_1.story
      * ...

Each of the `*.story` files is a single document consist of both the source and target texts. The structure of these files should be like following:

```markdown
First sentence.

Second sentence.

Third sentence (and so on).

@highlight

The abstract text.

```

### Sentence splitting and tokenization

Now it's time to preprocess your data using the Stanford CoreNLP library. Use the following commands to do this job for each train, validation, and test set.

```sh
python src/preprocess.py -mode tokenize -raw_path 'raw_data/train/stories' -save_path 'merged_stories_tokenized/train' -log_file 'logs/preprocess_train.log'
```

```sh
python src/preprocess.py -mode tokenize -raw_path 'raw_data/val/stories' -save_path 'merged_stories_tokenized/val' -log_file 'logs/preprocess_val.log'
```

```sh
python src/preprocess.py -mode tokenize -raw_path 'raw_data/test/stories' -save_path 'merged_stories_tokenized/test' -log_file 'logs/preprocess_test.log'
```

**Note:** You can run the following command to be sure that the number of output files is the same as the number of data you gave to the tokenizer.

```sh
find merged_stories_tokenized/train -maxdepth 1 -type f | wc -l
```

### Format to simpler JSON files

The next step is to convert the tokenized documents to JSON files.

```sh
python src/preprocess.py -mode custom_format_to_lines -raw_path 'merged_stories_tokenized/train' -save_path 'json_data/train' -n_cpus 1 -use_bert_basic_tokenizer false -map_path 'urls' -log_file 'logs/json_train.log'
```

```sh
python src/preprocess.py -mode custom_format_to_lines -raw_path 'merged_stories_tokenized/val' -save_path 'json_data/val' -n_cpus 1 -use_bert_basic_tokenizer false -map_path 'urls' -log_file 'logs/json_val.log'
```

```sh
python src/preprocess.py -mode custom_format_to_lines -raw_path 'merged_stories_tokenized/test' -save_path 'json_data/test' -n_cpus 1 -use_bert_basic_tokenizer false -map_path 'urls' -log_file 'logs/json_test.log'
```

### Format to PyTorch files

The following commands will convert the JSON files to PyTorch files using the *BertTokenizer* for all three sets (train, validation, test).

 ```sh
 python src/preprocess.py -mode custom_format_to_bert -dataset 'train' -raw_path 'json_data/train/' -save_path 'bert_data/train' -lower -n_cpus 1 -log_file 'logs/to_bert_train.log'
 ```

```sh
python src/preprocess.py -mode custom_format_to_bert -dataset 'val' -raw_path 'json_data/val/' -save_path 'bert_data/val' -lower -n_cpus 1 -log_file 'logs/to_bert_val.log'
```

```sh
python src/preprocess.py -mode custom_format_to_bert -dataset 'test' -raw_path 'json_data/test/' -save_path 'bert_data/test' -lower -n_cpus 1 -log_file 'logs/to_bert_test.log'
```

## Training

Now everything is ready to train the model.

This repository focuses on the abstractive setting of the PreSumm model, and it will use the BertAbs method. You need to apply some changes to the code if you are willing to train other kinds of models. The [CHANGES.md](https://github.com/parsa-abbasi/PreSumm/blob/master/CHANGES.md) file can be helpful to you in this case.

You can use a command like the following to set hyperparameters up and start training the model.

```sh
python src/train.py  -task abs -mode train -bert_data_path 'bert_data/train/' -dec_dropout 0.2 -model_path 'models' -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 10000 -batch_size 32 -train_steps 100000 -max_pos 512 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 8000 -warmup_steps_dec 4000 -visible_gpus 0 -log_file logs/abs_bert.log -large False -share_emb True -finetune_bert False -dec_layers 1 -enc_layers 1 -min_length 2
```

## Evaluation

### Validation

This step isn't necessarily required but it will be helpful to find which of the checkpoint models gives better results.

```sh
python src/train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path 'bert_data/val/' -log_file logs/val_abs_bert.log -model_path 'models' -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 256 -alpha 0.95 -min_length 1 -result_path 'results/val/results'
```

### Testing

Finally, the following command will produce the abstraction for the given texts and computes the ROUGE metrics using the produced texts and the true abstractions.

Don't forget to change the hyperparameters based on your task. You can define the model using the `test_from` argument.

```sh
python src/train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path 'bert_data/test/' -log_file logs/test_abs_bert.log -sep_optim true -use_interval true -visible_gpus 0 -alpha 0.95 -result_path 'results/test/results' -test_from 'models/model_step_100000.pt'
```

