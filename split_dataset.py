import pandas as pd

def split_data_into_binary_and_single_word_answers(csv_file, split):
    csv_data = pd.read_csv(csv_file, encoding='latin1', usecols=['file_name', 'QuestionText', 'answer' ])

    csv_data['answer'] = csv_data['answer'].str.upper()

    select = ['YES', 'NO']
    data_single_word = csv_data[~csv_data['answer'].isin(select)]
    data_binary = csv_data[csv_data['answer'].isin(select)]

    with open('metadata\\single_word_{}.csv'.format(split), 'wb') as f:
         data_single_word.to_csv(f, index=False)
    
    with open('metadata\\binary_{}.csv'.format(split), 'wb') as f:
         data_binary.to_csv(f, index=False)


csv_file_train = 'metadata\\clotho_aqa_train.csv'
csv_file_val = 'metadata\\clotho_aqa_train.csv'
csv_file_test = 'metadata\\clotho_aqa_train.csv'

split_data_into_binary_and_single_word_answers(csv_file_train, 'train')
split_data_into_binary_and_single_word_answers(csv_file_val, 'val')
split_data_into_binary_and_single_word_answers(csv_file_test, 'test')