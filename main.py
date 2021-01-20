import tensorflow_datasets as tfds
import tensorflow_datasets_ko.text.c4ko
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['gcs', 'local']:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(1)

    elif sys.argv[1] == 'gcs':
        tfds.load("c4ko/default", try_gcs=True)

    elif sys.argv[1] == 'local':
        tfds.load("c4ko/default", try_gcs=False)

    else:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(2)
