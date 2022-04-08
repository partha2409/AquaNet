import openl3
from conifg import data_config
import os


def main():
    dataset_path = data_config['data_dir']
    feat_dir = data_config['feat_dir']

    os.makedirs(feat_dir, exist_ok=True)
    audio_files = os.listdir(dataset_path)

    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(dataset_path, audio_file)
        openl3.process_audio_file(audio_path, output_dir=feat_dir, embedding_size=data_config['audio_embedding_size'], content_type='env')


if __name__ == '__main__':
    main()
