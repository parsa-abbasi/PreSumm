# Changes

## Data builder

**File path:** `src/prepro/data_builder.py`

1) The line 123 is changed from:

```python
command = ['java', '-cp', '/content/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
```

to:

```python
command = ['java', '-cp', 'stanford/stanford-corenlp-4.2.1/stanford-corenlp-4.2.1.jar', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
```

2) The line 134 is changed from:

```python
if num_orig != num_tokenized:
```

to:

```python
# I don't care about this exception
if (num_orig != num_tokenized) and False:
```

3) The line 210 is changed from:

```python
self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```

to:

```python
self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert_model')
```

4) The lines 447 and 448 is changed from:

```python
corpora = {'train': train_files}
    for corpus_type in ['train']:
```

to:

```python
print("The number of files:", len(train_files))
ctype = args.raw_path.split('/')[-1]
corpora = {ctype: train_files}
for corpus_type in [ctype]:
```

5) The line 456 and 466 is changed from:

```python
pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
```

to:

```python
pt_file = "{:s}/{:s}_{:d}.story.json".format(args.save_path, corpus_type, p_ct)
```

6) The line 482 is changed from:

```python
for json_f in glob.glob(args.raw_path + '*' + corpus_type + '.[0-9]*.json'):
```

to:

```python
for json_f in glob.glob(args.raw_path + '*' + corpus_type + '_[0-9]*.story.json'):
```

7) The following function is added:

```python
def custom_format_to_lines(args):
    corpus_mapping = {}
    train_files = []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        train_files.append(f)
    print("The number of files:", len(train_files))
    ctype = args.raw_path.split('/')[-1]
    corpora = {ctype: train_files}
    for corpus_type in [ctype]:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}_{:d}.story.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}_{:d}.story.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
```

8) The following function is added:

```python
def custom_format_to_bert(args):
        if (args.dataset != ''):
            datasets = [args.dataset]
            print('dataset')
        else:
            datasets = ['train']
        for corpus_type in datasets:
            a_lst = []
            print('.' + corpus_type + '.0.json')
            for json_f in glob.glob(args.raw_path + '*' + corpus_type + '_[0-9]*.story.json'):
                print(json_f)
                real_name = json_f.split('/')[-1]
                print(real_name)
                a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
            print(a_lst)
            pool = Pool(args.n_cpus)
            for d in pool.imap(_format_to_bert, a_lst):
                pass

            pool.close()
            pool.join()
```

## Train abstractive

**File path:** `src/train_abstractive.py`

1) The lines 189, 221, 249, and 325 is changed from:

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
```

to:

```python
tokenizer = BertTokenizer.from_pretrained('bert_model', cache_dir=args.temp_dir)
```

2) The line 218 of the `src/train_abstractive.py` file is changed from:

```python
test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                   args.test_batch_size, device,
                                   shuffle=False, is_test=True)
```

to:

```python
c_type = 'test'
    if args.mode == 'validate':
        c_type = 'valid'

test_iter = data_loader.Dataloader(args, load_dataset(args, c_type, shuffle=False),
                                   args.test_batch_size, device,
                                   shuffle=False, is_test=True)
```

## Model builder

**File path:** `src/models/model_builder.py`

1) The lines 118-121 is changed from:

```python
if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
```

to:

```python
self.model = BertModel.from_pretrained('bert_model', cache_dir=temp_dir)
```

## Data loader

**File path:** `src/models/data_loader.py`

1) The following code is added before the line 75:

```python
if corpus_type == 'valid':
        corpus_type = 'val'
```

2) The line 84 is changed from:

```python
pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
```

to:

```python
pts = sorted(glob.glob(args.bert_data_path + corpus_type + '_[0-9]*.story.bert.pt'))
```

3) The line 93 is changed from:

```python
pt = args.bert_data_path + '.' + corpus_type + '.pt'
```

to:

```python
pt = args.bert_data_path + corpus_type + '.story.bert.pt'
```

## Preprocess

**File path:** `src/models/data_loader.py`

1) The line 64 is changed from:

```python
parser.add_argument('-log_file', default='../../logs/cnndm.log')
```

to:

```python
parser.add_argument('-log_file', default='logs/cnndm.log')
```

