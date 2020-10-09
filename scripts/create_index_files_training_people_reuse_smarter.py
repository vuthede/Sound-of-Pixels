import os
import glob
import argparse
import random
import fnmatch
import pandas as pd
from  tqdm import tqdm 
import concurrent.futures



def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    prefix = "/media/Databases/preprocess_avspeech/segment_clean_sam_ffmpeg_small"
    parser.add_argument('--root_audio', default=f'{prefix}/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default=f'{prefix}/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    parser.add_argument('--reuse_old_valset', default=1, type=int,
                        help="Reuse old valset or split again")
    parser.add_argument('--oldvalset', default='./data/val.csv', type=str,
                        help="csv file contain old valid files")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    
    # Get unique video for train/ val

    video_ids = os.listdir(args.root_audio)
    print("All video ids len :", len(video_ids))

    if args.reuse_old_valset ==0: # Create val again
        print("randomly spliet speaker again!!")
        n_video_train = int(len(video_ids) * 0.9)
        random.shuffle(video_ids)
        video_train_ids = video_ids[0:n_video_train]
        video_valid_ids = video_ids[n_video_train:]
    else:
        print("Keep reusing old valid set with additional training set")
        if not os.path.isfile(args.oldvalset):
            raise FileNotFoundError

        video_valid_ids = list(pd.read_csv(args.oldvalset, header=None)[0]) #paths
        video_valid_ids = [i.split("/")[-2] for i in video_valid_ids] # ids
        video_train_ids = list(set(video_ids) - set(video_valid_ids))

    print("Len trains id: ", len(video_train_ids))
    print("Len valid id: ", len(video_valid_ids))

    
    
    # audio_files = find_recursive(args.root_audio, ext='.mp3')
    info_train = []
    info_val = []

    def thread_function_train(video_id):
        audio_paths = glob.glob(f'{args.root_audio}/{video_id}/*.wav')
        for p in audio_paths:
            v = p.replace(args.root_audio, args.root_frame).replace('.wav', '.mp4')
            frame_files = os.listdir(v)
            if len(frame_files)>=10*3:
                info_train.append(','.join([p, v, str(len(frame_files))]))
    
    def thread_function_test(video_id):
        audio_paths = glob.glob(f'{args.root_audio}/{video_id}/*.wav')
        for p in audio_paths:
            v = p.replace(args.root_audio, args.root_frame).replace('.wav', '.mp4')
            frame_files =  os.listdir(v)
            
            if len(frame_files)>=10*3:
                info_val.append(','.join([p, v, str(len(frame_files))]))


    print("Create training info.........")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(thread_function_train, video_train_ids), total=len(video_train_ids)))

    # for i in tqdm(video_train_ids):
    #     audio_paths = glob.glob(f'{args.root_audio}/{i}/*.wav')
    #     for p in audio_paths:
    #         v = p.replace(args.root_audio, args.root_frame).replace('.wav', '.mp4')
    #         frame_files = glob.glob(v + '/*.jpg')
    #         info_train.append(','.join([p, v, str(len(frame_files))]))

    print("Create valid info.........")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(thread_function_test, video_valid_ids), total=len(video_valid_ids)))


    # for i in tqdm(video_valid_ids):
    #     audio_paths = glob.glob(f'{args.root_audio}/{i}/*.wav')
    #     for p in audio_paths:
    #         v = p.replace(args.root_audio, args.root_frame).replace('.wav', '.mp4')
    #         frame_files = glob.glob(v + '/*.jpg')
    #         info_val.append(','.join([p, v, str(len(frame_files))]))

    # for audio_path in audio_files:
    #     frame_path = audio_path.replace(args.root_audio, args.root_frame) \
    #                            .replace('.mp3', '.mp4')
    #     frame_files = glob.glob(frame_path + '/*.jpg')
        

    #     if len(frame_files) >= 30 * 3: #90 frames
    #         file_id = audio_path.split("/")[-2]
    #         if file_id in video_train_ids:
    #             info_train.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    #         else:
    #             info_val.append(','.join([audio_path, frame_path, str(len(frame_files))]))
                
            
    print('{} audio/frames train pairs found.'.format(len(info_train)))
    print('{} audio/frames val pairs found.'.format(len(info_val)))


    # split train/val
    trainset = info_train
    valset = info_val
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
