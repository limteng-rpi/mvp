Dependencies
============
- Python 3.5+
Python packages:
- Pytorch 0.3.X (The code may not work with PyTorch 0.4.X as some APIs are different.)
- Requests (to send entity linking requests)
- NLTK (tokenization, POS tagging)

Files
=====
- `pipeline_baseline.py`: pipeline for the baseline model without MFD and
 background knowledge features.
- `pipeline_mfd.py`: pipeline for the model with MFD features.
- `pipeline_mfd_bk.py`: pipeline for the model with MFD and background 
 knowledge features.
- `tagme.py`: run entity linking on the data set using TagMe.
- `data_processing.py`: contains some functions to clean and enrich the entity
 linking results (optional).
 
File Format
===========
The tweet data set should be encoded in 3-column TSV files as follows:
```
tag:search.twitter.com,2005:592839674642718722    God bless the clergy #BaltimoreRiots    CH
tag:search.twitter.com,2005:592840383434006529    Shame on these racist white women for trying to steal this poor black man's purse : #BaltimoreRiots     LB
tag:search.twitter.com,2005:592840285736050688    Nothing screams justice like destroying your own town and stealing innocent people's property #BaltimoreRiots   AS,LB
...
```
 
How to Run
==========

Entity Linking
--------------
Command:
```
python tagme.py -i <INPUT_FILE> -o <OUTPUT_FILE> -t <TOKEN>
```
- -i, --input: Path to the TSV format tweet data set
- -o, --output: Path to the output file (TagMe result)
- -t, --token: TagMe application token. To use the TagMe RESTful API, you need
 to register an account and get a free token (see:
 https://services.d4science.org/web/tagme/tagme-help)

Model Training
--------------
Command:
```
python pipeline_baseline/baseline_mfd/baseline_mfd_bk.py --train <TRAIN_SET_FILE>
 --dev <DEV_SET_FILE>
 --test <TEST_SET_FILE>
 --mode train
 --el <TAGME_RESULT_FILE>
 --mfd <MORAL_FOUNDATION_DICT_FILE>
 --model <MODEL_OUTPUT_DIR>
 --output <TEST_SET_RESULT_OUTPUT_FILE>
 --gpu 1
 --embedding <PATH_TO_WORD_EMBEDDING_FILE_FOR_TWEETS>
 --el_embedding <PATH_TO_WORD_EMBEEDING_FILE_FOR_BACKGROUND_KNOWLEDGE>
```
- --train,--dev,--test: Training/dev/test set files
- --el: Path to the TagMe result file.
- --mfd: Path to the Moral Foundation Dictionary file.
- --model: Path to the model output directory.
- --output: Path to the output file. At the end of the training, the model will
 be applied to the test set, and the results will be written to this file.
- -m, --mode: Mode, currently only 'train' is implement. If you want to predict
 moral values on an unlabeled data set, label all instances as 'NM' (non-moral),
 the prediction results will be written to the output file (see `--output`).
- --labels: Label set (default: CH,FC,AS,LB,PD).
- --learning_rate: Learning rate (default=0.005).
- --batch_size: Batch size (default=20).
- --max_epoch: Max training epoch number (default=30).
- --max_seq_len: Max sequence length (tweets longer than this length will be
 truncated)
- --embedding: Path to the embedding file for tweets.
- --el_embedding: Path to the embedding file for background knowledge. Note that
 we use different pre-trained word embedding for tweets and background knowledge.
- --embedding_dim: Word embedding dimension (default=100). Should be consistent 
 with embeddings in the `--embedding` file.
- --el_embedding_dim: Word embedding dimension (default=100). Should be 
 consistent with embeddings in the `--el_embedding` file.
- --hidden_size: LSTM hidden state size (default=100).
- --linear_size: Sizes of hidden layers that process the LSTM output (default=50).
- --el_linear_sizes: Sizes of hidden layers that process the background knowledge
 feature vector (default=5).
- --mfd_linear_sizes: Sizes of hidden layers that process the MFD feature vector.
 (default=5).
 
- --gpu: Use GPU or not (default=1).
- --device: Select GPU (if you have multiple GPUs on your machine).