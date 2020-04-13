import codecs
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def preprocessing_paths(video_paths, path2videos, type_='jpg'):
    paths = []
    for video_path in video_paths:
        folder = video_path.split('.')[-1]
        with codecs.getreader("utf-8")(tf.io.gfile.GFile(video_path, "rb")) as f:
            for path in f:
                new_path = path.replace('<PATH_TO_EXTRACTED_AND_RESIZED_FRAMES>', path2videos).strip()
                if len(new_path) > 1:
                    paths.append([new_path+type_, folder.lower()])
    print('Completed')
    return paths

def preprocessing_sentences(paths_sentences, max_len=55):
    outputs = []
    tokenizer = Tokenizer(oov_token='<unk>', filters='!"#$%&()*+,-:;=?@[\\]^_`{|}~\t\n')
    for path in paths_sentences:
        sentences = list()
        if path.split('.')[-1] == 'train':  
            print('Reading training file ...')  
            with codecs.getreader("utf-8")(tf.io.gfile.GFile(path, "rb")) as f:
                for sentence in f:
                    sentences.append('<s> ' + sentence.strip() + ' </s>')
                print('training on train sentences to make the vocab ...')
                tokenizer.fit_on_texts(sentences)
                tokenizer.word_index['<pad>'] = 0
                tokenizer.index_word[0] = '<pad>'
                print('training text to index sequences ..')
                sequences_token = tokenizer.texts_to_sequences(sentences)
                print('training index sequences to padded sequences ..')
                sequences_padded = pad_sequences(sequences_token, maxlen=max_len, padding='post')
                outputs.append(sequences_padded)
        else:
            print('Computing same steps over dev and test sequences')
            with codecs.getreader("utf-8")(tf.io.gfile.GFile(path, "rb")) as f:
                for sentence in f:
                    sentences.append('<s> ' + sentence.strip() + ' </s>')  
            sequences_token = tokenizer.texts_to_sequences(sentences)
            sequences_padded = pad_sequences(sequences_token, maxlen=max_len, padding='post')
            outputs.append(sequences_padded)         
    print('Completed')
    return (outputs, tokenizer)

def table_paths_dataset(preprocessed_paths, preprocessed_sentences):
    print('Creating the table paths for flow_from_tablePaths ..')
    padded_sentences = preprocessed_sentences[0]
    for i in range(1, len(preprocessed_sentences)):
        padded_sentences = np.concatenate([padded_sentences, preprocessed_sentences[i]], axis=0)
    pre_dataset = []
    for i in range(padded_sentences.shape[0]):
        pre_dataset.append([preprocessed_paths[i][0],
                            preprocessed_paths[i][1],
                            str(padded_sentences[i].tolist())[1:-1]
                           ]
                          )
    return np.r_[pre_dataset]# Los archivos no deben tener enter al final