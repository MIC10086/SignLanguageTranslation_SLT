import tensorflow as tf
from .data_augmentation import frame_sampling

def build_datasets(table_paths, args):
    raw_data = flow_from_tablePaths(table_paths, lambda x: x, args.inputShape[1:3])
    train_gen = raw_data.data_generator(1, args.inputShape[-1])
    test_gen = raw_data.data_generator(2, args.inputShape[-1])
    try:
        dev_gen = raw_data.data_generator(3, args.inputShape[-1])
    except:
        dev_gen = None
    
    def train_gen_sampling():
    for v, l in train_gen:
        for new_v in frame_sampling(v, args.inputShape[0]):
            yield new_v, l
            
    def test_gen_sampling():
        for v, l in test_gen:
            for new_v in frame_sampling(v, args.inputShape[0]):
                yield new_v, l
    if dev_gen:
        def dev_gen_sampling():
            for v, l in dev_gen:
                for new_v in frame_sampling(v, args.inputShape[0]):
                    yield new_v, l
    