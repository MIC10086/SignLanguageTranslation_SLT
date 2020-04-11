import tensorflow as tf
from DatasetsLoaderUtils import flow_from_tablePaths
from data_augmentation import frame_sampling

def build_datasets(table_paths, video_shape):
    raw_data = flow_from_tablePaths(table_paths, lambda x: x, video_shape[1:3])