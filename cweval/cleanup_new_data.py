import os
import shutil
from commons import BENCHMARK_DIR
import argparse

def cleanup_new_data_classes(name_data):
    short_name = name_data[:2]
    new_lang_dir = os.path.join(BENCHMARK_DIR, 'core', short_name)
    if os.path.exists(new_lang_dir):
        shutil.rmtree(new_lang_dir)
        print(f'Removed temporary language directory {new_lang_dir}')

def main():
    # take the first argument as a comma separated list of names or paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Path to the new data class')
    args = parser.parse_args()
    names_data = args.datasets.split(',')
    # Read the cleanup languages from the file
    with open(names_data[0] + '_cleanup.txt', 'r') as f:
        cleanup_languages = f.read().splitlines()
    for lang in cleanup_languages:
        cleanup_new_data_classes(lang)

if __name__ == "__main__":
    main()