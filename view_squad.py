from pathlib import Path

import tensorflow_datasets as tfds


def view_squad_v11():
    data, info = tfds.load("squad/v1.1", data_dir="/home/chris/tensorflow_datasets", with_info=True)
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
                    'answers': x['answers']['text'].numpy()[0].decode("utf-8"),
                    'context': x['context'].numpy().decode("utf-8"),
                    'id': x['id'].numpy().decode("utf-8"),
                    'question': x['question'].numpy().decode("utf-8"),
                    'title': x['title'].numpy().decode("utf-8"),
                }
                print(example)
                exit(1)


if __name__ == '__main__':
    view_squad_v11()
