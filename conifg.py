data_config = {
    'train_metadata_path': '../metadata/binary_train.csv',  # CSV containing audio URLs, Questions, Answers,filenames
    'val_metadata_path': '../metadata/binary_val.csv',
    'test_metadata_path': '../metadata/binary_test.csv',
    'data_dir': '../dataset/audio_files',  # path to store downloaded data
    'feat_dir': '../dataset/features',
    'pre_trained_word_embeddings_file': '../wiki-news-300d-1M.vec',
    'audio_embedding_size': 512
}

model_config = {
    # general params
    'net_type': 'aquanet',
    'output_dir': '.',

    # learning params
    'learning_rate': 0.001,
    'batch_size': 1,
    'num_workers': 8,
    'num_epochs': 50,

    # audio network
    'audio_input_size': data_config['audio_embedding_size'],
    'audio_lstm_n_layers': 2,
    'audio_lstm_hidden_size': 128,
    'audio_bidirectional': True,
    'audio_lstm_dropout': 0.2,


    # NLP network
    'text_input_size': 300,  # pretrained embedding size from fasttext
    'text_lstm_n_layers': 2,
    'text_lstm_hidden_size': 128,
    'text_bidirectional': True,
    'text_lstm_dropout': 0.2,

    # classification
    'n_dense1_units': 256,
    'n_dense2_units': 128,
    'n_classes': 1  # To be changed based on answers
}

dense1_input = 0
if model_config['audio_bidirectional']:
    dense1_input = dense1_input + 2 * model_config['audio_lstm_hidden_size']
else:
    dense1_input = dense1_input + model_config['audio_lstm_hidden_size']

if model_config['text_bidirectional']:
    dense1_input = dense1_input + 2 * model_config['text_lstm_hidden_size']
else:
    dense1_input = dense1_input + model_config['text_lstm_hidden_size']

model_config['dense1_input'] = dense1_input