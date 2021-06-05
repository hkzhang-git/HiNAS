from utils import (sid_dict_build, json_save, make_if_not_exist)
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='dataset preprocess')
    parser.add_argument('--data_root', type=str, default='D:/02_data/nas_data')
    parser.add_argument('--task', type=str, default='sid')
    parser.add_argument('--dataset', type=str, default='Sony')
    parser.add_argument('--json_dir', type=str, default='dataset_json')
    args = parser.parse_args()

    train_dict, test_dict = sid_dict_build(args)

    json_save_dir = os.path.join(args.json_dir, args.task, args.dataset)
    make_if_not_exist(json_save_dir)
    json_save(os.path.join(json_save_dir, 'train.json'), train_dict)
    json_save(os.path.join(json_save_dir, 'test.json'), test_dict)


if __name__ == '__main__':
    main()





