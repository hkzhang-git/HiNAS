from utils import (sr_dict_build, json_save, make_if_not_exist)
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='dataset preprocess')
    parser.add_argument('--data_root', type=str, default='D:/02_data/nas_data')
    parser.add_argument('--task', type=str, default='sr')
    parser.add_argument('--datasets', type=str, default=['DIV2K_800', 'Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109'])
    parser.add_argument('--json_dir', type=str, default='dataset_json')
    args = parser.parse_args()


    dict_list = sr_dict_build(args)

    json_save_dir = os.path.join(args.json_dir, args.task)
    make_if_not_exist(json_save_dir)
    for dataset, dict in zip(args.datasets, dict_list):
        json_save(os.path.join(json_save_dir, '{}.json'.format(dataset)), dict)

if __name__ == '__main__':
    main()




