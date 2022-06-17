# AquaNet
This repository contains a sample code to show the usage of Clotho-AQA audio question answering dataset. The Clotho-AQA is a new audio question answering dataset consisting of 1991 audio files and 6 different questions per each file. Each question is answered independently by 3 different annotators. Complete details about the dataset can be found in our paper available online at https://arxiv.org/pdf/2204.09634.pdf. If you use our dataset, please cite our paper.

# Downloading the Clotho-AQA dataset
Download the clotho-AQA dataset from https://zenodo.org/record/6473207. Extract the zip file and place the audio files in `dataset/audio_files/` directory. Place the csv files in `metadata/` directory. 
# Extracting features
In this baseline model, we use the openL3 open source python library to computing deep audio embeddings. To compute the embeddings first install openL3 using
`pip install openl3`.

Now, run `extract_features.py` . This may take a while to compute the embeddings for all the audio files in the dataset. Once it is complete, you should be able to find the deep audio embeddings stored in `dataset/features` 

We also use Fasttext word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset to create word embeddings for our questions. This can be downloaded from https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip 
# Training the model
First, run `split_dataset.py` to generate csv files for binary answers and single word answers.

To train the model run `train.py`. The model checkpoint will be saved for every 10 epochs. If you want to continue training from a saved checkpoint, assign the checkpoint path to `pre_trained_model_path` variable in `train.py`
# Inference
Once the model is trained, update the variables `model_dir` and `model_path` in `run_inference.py` and execute the file.
