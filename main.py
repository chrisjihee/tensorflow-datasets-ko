import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset

if __name__ == '__main__':
    # print('가')
    # print(type('가'.encode('utf-8')))
    #
    # exit(1)
    if len(sys.argv) < 2 or sys.argv[1] not in ['gcs', 'local']:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(1)

    elif sys.argv[1] == 'gcs':
        tfds.load("c4ko/default", try_gcs=True)

    elif sys.argv[1] == 'local':
        x, i = tfds.load("c4ko/default", with_info=True)
        # print(f'i={i}')
        print(f'x={x.keys()}')
        dataset: PrefetchDataset = x['train']
        # dataset: PrefetchDataset = x['validation']
        for data in list(dataset):
            print('-' * 120)
            print(data['text'].numpy().decode("utf-8"))

    else:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(2)
