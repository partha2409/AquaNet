# AquaNet
This repository contains a sample code to show the usage of Clotho-AQA audio question answering dataset. The Clotho-AQA is a new audio question answering dataset consisting of 1991 audio files and 6 different questions per each file. Each question is answered independently by 3 different annotators. Complete details about the dataset can be found in our paper available online at 'arXiv link'.

If you use our dataset, please consider citing our paper.

# Downloading the Clotho-AQA dataset
Download the clotho-AQA dataset from 'Zenodo link'. Extract the zip file and place the audio files in `dataset\audio_files` directory. Place the csv files in `metadata` directory 
# Extracting features
In this baseline model, we use the openL3 open source python library to computing deep audio embeddings. TO compute the embeddings first install openL3 using
`pip install openl3`

Now, run the `extract_embeddings.py` . This may take a few minutes to compute the embeddins for all the files. Once it is complete, you should be able to find the deep audio embeddings stored in `dataset\features` 

We also use Fasttext word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset to create word embeddings for our questions. This can be downloaded from https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip 
# Training the model
# Inference

