import json
from pathlib import Path

import tensorflow_datasets as tfds

from tensorflow_datasets_ko.text.c4ko import C4ko


def view_c4ko_default():
    print(f"[c4ko] {C4ko.BUILDER_CONFIGS[0].description}")
    data, info = tfds.load("c4ko/default", data_dir="/home/chris/tensorflow_datasets", with_info=True)
    print(info)
    print('-' * 120)

    num_all_example, num_kor_example = 0, 0
    with Path("output/c4ko.txt").open("w") as out:
        for key in data.keys():
            for x in data[key]:
                print(f"key={key}")
                print(x.keys())
                num_all_example += 1
                example = {
                    'text': x['text'].numpy().decode("utf-8"),
                    'content-length': x['content-length'].numpy().decode("utf-8"),
                    'content-type': x['content-type'].numpy().decode("utf-8"),
                    'timestamp': x['timestamp'].numpy().decode("utf-8"),
                    'url': x['url'].numpy().decode("utf-8")
                }
                print(example)
                exit(1)
        print('-' * 120)
        print(json.dumps({'num_all_example': num_all_example, 'num_kor_example': num_kor_example, 'rate_kor_example': num_kor_example / num_all_example}, indent=4))


if __name__ == '__main__':
    view_c4ko_default()
