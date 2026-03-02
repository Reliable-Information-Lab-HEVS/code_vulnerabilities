import argparse
import os
from commons import BENCHMARK_DIR, LANGS
from pathlib import Path
import shutil

def add_test_cases(language, path_to_save):
    lang_dir = os.path.join(BENCHMARK_DIR, '..', 'backup', 'core_backup', language)
    if not os.path.exists(lang_dir):
        raise ValueError(f'Language directory {lang_dir} does not exist.')
    # iterate over the files, if it ends with _test.py, copy it to path_to_save
    for file_name in os.listdir(lang_dir):
        if file_name.endswith('_test.py'):
            full_file_name = os.path.join(lang_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copyfile(full_file_name, os.path.join(path_to_save, file_name))
        if language in ['c', 'cpp', 'js', 'go']:
            if file_name.endswith('_unsafe.' + language):
                full_file_name = os.path.join(lang_dir, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copyfile(full_file_name, os.path.join(path_to_save, file_name))
            


def handle_new_data_classes(names_data):
    new_langs = []
    cleanup_languages = []
    for name_data in names_data:
        if name_data in LANGS:
            new_langs.append(name_data)
        else:
            #it's a path, let's first find the name of the dataset, which is the part after the last /
            # and then take the first two characters as the language code
            dataset_name = name_data.split('/')[-1]
            short_name = dataset_name.split('_')[0]
            if short_name not in LANGS:
                raise ValueError(f'Language {name_data} not supported, make sure the first two characters correspond to a supported language, one of {LANGS}')
            else:
                target_dir = os.path.join(BENCHMARK_DIR, 'core', short_name)
                if os.path.exists(target_dir):
                    raise ValueError(f'Target directory {target_dir} already exists, please remove it first.')
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                # copy test cases from backup
                add_test_cases(short_name, target_dir)
                new_langs.append(short_name)
                cleanup_languages.append(short_name)
                # Now copy all files from the specified path to the target_dir, the specified path is relative to cwd
                cwd = os.getcwd()
                print(f'Copying new data class from {name_data} to {target_dir}')
                src_lang_dir = os.path.join(cwd, name_data)
                if not os.path.exists(src_lang_dir):
                    raise ValueError(f'Source language directory {src_lang_dir} does not exist.')
                for file_name in os.listdir(src_lang_dir):
                    full_file_name = os.path.join(src_lang_dir, file_name)
                    #get Path object
                    path_obj = Path(full_file_name)
                    path_new_obj = Path(target_dir) / file_name
                    if os.path.isfile(full_file_name):
                        shutil.copyfile(full_file_name, target_dir + '/' + file_name)
                
    return new_langs, cleanup_languages


def main():
    # take the first argument as a comma separated list of names or paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Path to the new data class')
    args = parser.parse_args()
    names_data = args.datasets.split(',')
    
    new_langs, cleanup_languages = handle_new_data_classes(names_data)
    print(f'Using languages: {new_langs}')
    #Now write the cleanup languages to a file
    with open(names_data[0] + '_cleanup.txt', 'w') as f:
        for lang in cleanup_languages:
            f.write(lang + '\n')

if __name__ == "__main__":
    main()