import tensorflow as tf
from .DatasetsLoaderUtils import flow_from_tablePaths
from .data_augmentation import frame_sampling

def build_datasets(table_paths, args):
    raw_data = flow_from_tablePaths(table_paths, lambda x: x, args.inputShape[1:3])

    train_gen = raw_data.data_generator(1, args.inputShape[-1])
    def train_gen_sampling():
        for v, l in train_gen:
            for new_v in frame_sampling(v, args.inputShape[0]):
                yield new_v, l
    train_data = tf.data.Dataset.from_generator(train_gen_sampling, (tf.float32, tf.int64),
        (args.inputShape, [])
        )

    test_gen = raw_data.data_generator(2, args.inputShape[-1])
    def test_gen_sampling():
        for v, l in test_gen:
            for new_v in frame_sampling(v, args.inputShape[0]):
                yield new_v, l
    test_data = tf.data.Dataset.from_generator(test_gen_sampling, (tf.float32, tf.int64),
        (args.inputShape, [])
        )

    try:
        dev_gen = raw_data.data_generator(3, args.inputShape[-1])
        def dev_gen_sampling():
            for v, l in dev_gen:
                for new_v in frame_sampling(v, args.inputShape[0]):
                    yield new_v, l
        dev_data = tf.data.Dataset.from_generator(dev_gen_sampling, (tf.float32, tf.int64),
            (args.inputShape, [])
            )
    except:
        dev_data = None
    
    return train_data, test_data, dev_data