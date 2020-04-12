import tensorflow as tf
import numpy as np
from .DatasetsLoaderUtils import flow_from_tablePaths
from .data_augmentation import frame_sampling

def build_datasets(table_paths, args):
    raw_data = flow_from_tablePaths(table_paths, lambda x: x, args.inputShape[1:3])

    train_gen = raw_data.data_generator(1, args.inputShape[-1])
    def train_gen_sampling():
        for v, l in train_gen:
            s = np.r_[[int(j) for j in (raw_data.to_class[l]).split(", ")]]
            for new_v in frame_sampling(v, args.inputShape[0]):
                yield (new_v, s[:-1]), s[1:]
    train_data = tf.data.Dataset.from_generator(train_gen_sampling, ((tf.float32, tf.int64), tf.int64))

    test_gen = raw_data.data_generator(2, args.inputShape[-1])
    def test_gen_sampling():
        for v, l in test_gen:
            s = np.r_[[int(j) for j in (raw_data.to_class[l]).split(", ")]]
            for new_v in frame_sampling(v, args.inputShape[0]):
                yield (new_v, s[:-1]), s[1:]
    test_data = tf.data.Dataset.from_generator(test_gen_sampling, ((tf.float32, tf.int64), tf.int64))

    try:
        dev_gen = raw_data.data_generator(3, args.inputShape[-1])
        def dev_gen_sampling():
            for v, l in dev_gen:
                s = np.r_[[int(j) for j in (raw_data.to_class[l]).split(", ")]]
                for new_v in frame_sampling(v, args.inputShape[0]):
                    yield (new_v, s[:-1]), s[1:]
        dev_data = tf.data.Dataset.from_generator(dev_gen_sampling, ((tf.float32, tf.int64), tf.int64))
    except:
        dev_data = None
    
    return train_data, test_data, dev_data