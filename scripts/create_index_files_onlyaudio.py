import os
import glob
import random


def write_all_ids(ids, txt):
    with open(txt, 'w') as f:
        for i in ids:
            f.write(str(i))
            f.write("\n")

def read_all_ids(txt):
    with open(txt, 'r') as f:
        ids = f.readlines()
        ids =  [i.strip() for i in ids]
        return ids

def split_data_set_by_speaker(data_dir,validsettxt="valid.txt", cache_valid=True):
    """
    \data_dir is the folder contains all audio files with format `trim_audio_train_<string>_<number>.wav`
    """
    files = os.listdir(data_dir)
    files = sorted(files)

    if os.path.isfile(validsettxt):
        print(f"Using {validsettxt} as valid file")
        valid_files = read_all_ids(validsettxt)
    else:
        print(f"{validsettxt} does not exist. So we will generate new valid file")
        s = random.randint(0, len(files)-200)
        valid_files = files[s:s+200]
        write_all_ids(valid_files,"valid.txt")

    
    train_files = list(set(files) - set(valid_files)) 
    

    return train_files, valid_files



if __name__ == '__main__':
    audio_dir = "/home/vtde/traindata"
    output = "../data"
   

    train_files, valid_files = split_data_set_by_speaker(data_dir=audio_dir,validsettxt="valid.txt")

    print(f'Len trainfiles: {len(train_files)}, len validfiles :{len(valid_files)}')


    # split train/val
    for name, files in zip(['train', 'val'], [train_files, valid_files]):
        filename = '{}.csv'.format(os.path.join(output, name))
        with open(filename, 'w') as f:
            for file in files:
                line = f'{audio_dir}/{file},None,-1'
                f.write(line + '\n')
        print('{} items saved to {}.'.format(len(files), filename))

    print('Done!')
