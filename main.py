import json
import re
import sys
from pathlib import Path

import tensorflow_datasets as tfds

from tensorflow_datasets_ko.text.c4ko import C4ko

from soynlp.normalizer import repeat_normalize

rex_useless = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')
rex_website = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')


def clean(x):
    x = rex_useless.sub(' ', x)
    x = rex_website.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


def check_c4ko_dataset(rate_kor_letters=0.3):
    data, info = tfds.load("c4ko/default", data_dir="/home/chris/tensorflow_datasets", with_info=True)
    print(f"[c4ko] {C4ko.BUILDER_CONFIGS[0].description}")
    print(info)
    print('-' * 120)

    num_all_example, num_kor_example = 0, 0
    with Path("output/c4ko.txt").open("w") as out:
        for key in data.keys():
            for x in data[key]:
                num_all_example += 1
                example = {
                    'text': x['text'].numpy().decode("utf-8"),
                    'content-length': x['content-length'].numpy().decode("utf-8"),
                    'content-type': x['content-type'].numpy().decode("utf-8"),
                    'timestamp': x['timestamp'].numpy().decode("utf-8"),
                    'url': x['url'].numpy().decode("utf-8")
                }
                if re.search('[가-힣ㄱ-ㅎㅏ-ㅣ]', example['text']):
                    example['len_text'] = len(example['text'])
                    example['num_all_letters'] = len(re.findall(r'\w', example['text']))
                    example['num_kor_letters'] = len(re.findall('[가-힣ㄱ-ㅎㅏ-ㅣ]', example['text']))
                    example['rate_kor_letters'] = example['num_kor_letters'] / example['num_all_letters']

                    if example['rate_kor_letters'] >= rate_kor_letters:
                        num_kor_example += 1

                        example['text_clean'] = clean(example['text'])
                        example['len_text_clean'] = len(example['text_clean'])
                        # example['num_all_letters_clean'] = len(re.findall(r'\w', example['text_clean']))
                        # example['num_kor_letters_clean'] = len(re.findall('[가-힣ㄱ-ㅎㅏ-ㅣ]', example['text_clean']))
                        # example['rate_kor_letters_clean'] = example['num_kor_letters'] / example['num_all_letters']
                        example['rate_text_clean'] = example['len_text_clean'] / example['len_text']

                        if example['rate_text_clean'] <= 0.7:
                            print('=' * 80)
                            print(example['url'])

                            pre_validity = 1
                            num_cont, num_cut = 0, 0
                            valid_lines = []
                            clean_rates = []
                            korean_rates = []
                            for i, line in enumerate(example['text'].split('\n')):
                                original = line.strip()
                                cleaned = clean(line.strip())
                                clean_rate = int((1.0 - len(cleaned) / len(original)) * 100.0)
                                korean_rate = 0 if len(re.findall(r'\w', cleaned)) == 0 else int(len(re.findall('[가-힣ㄱ-ㅎㅏ-ㅣ]', cleaned)) / len(re.findall(r'\w', cleaned)) * 100.0)
                                cur_validity = 0 if clean_rate >= 70 or korean_rate <= 30 else 1
                                if cur_validity == 1:
                                    valid_lines.append(cleaned)
                                if pre_validity >= 0 and pre_validity != cur_validity:
                                    num_cut += 1
                                elif pre_validity == 1 and cur_validity == 1:
                                    num_cont += 1
                                clean_rates.append(clean_rate)
                                korean_rates.append(korean_rate)
                                print(f"i={i} / clean_rate={clean_rate}% / korean_rate={korean_rate}% / line_type={cur_validity} / num_cont={num_cont} / num_cut={num_cut} / cleaned={cleaned} / original={original}")
                                pre_validity = cur_validity
                            avg_clean_rate = int(sum(clean_rates) / len(clean_rates))
                            avg_korean_rate = int(sum(korean_rates) / len(korean_rates))
                            if len(valid_lines) >= 1 and avg_clean_rate <= 50 and avg_korean_rate >= 50:
                                print('=' * 80)
                                for line in valid_lines:
                                    print(line)
                                print(f"avg_clean_rate={avg_clean_rate} / avg_korean_rate={avg_korean_rate} / num_cont={num_cont} / num_cut={num_cut} / num_valid={len(valid_lines)}")
                            print('=' * 80)
                            print()

                        # if example['rate_text_clean'] > 0.3:
                        #     example.pop('text')
                        # else:
                        #     example.pop('text_clean')
                        if num_kor_example % 1000 == 0:
                            print(json.dumps({'num_all_example': num_all_example, 'num_kor_example': num_kor_example, 'rate_kor_example': num_kor_example / num_all_example}))
                        out.write(json.dumps(example, ensure_ascii=False, indent=4) + '\n' + '-' * 10 + '\n')
        print('-' * 120)
        print(json.dumps({'num_all_example': num_all_example, 'num_kor_example': num_kor_example, 'rate_kor_example': num_kor_example / num_all_example}, indent=4))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['gcs', 'local']:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(1)

    elif sys.argv[1] == 'gcs':
        tfds.load("c4ko/default", try_gcs=True)

    elif sys.argv[1] == 'local':
        check_c4ko_dataset()

    else:
        print("usage: python main.py [by]")
        print("       - by: 'gcs' or 'local'")
        exit(2)
