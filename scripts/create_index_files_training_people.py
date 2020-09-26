import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    prefix = "/media/Databases/preprocess_avspeech/segment_sam_with_noise"
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
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.mp3')
    
    # Get unique video for train/ val
    video_ids = os.listdir(args.root_audio)
    n_video_train = int(len(video_ids) * 0.9)
    random.shuffle(video_ids)

    video_train_ids = video_ids[0:n_video_train]
    video_valid_ids = video_ids[n_video_train:]

    info_train = []
    info_val = []

    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        

        if len(frame_files) >= 30 * 3: #90 frames
            file_id = audio_path.split("/")[-2]
            if file_id in video_train_ids:
                info_train.append(','.join([audio_path, frame_path, str(len(frame_files))]))
            else:
                info_val.append(','.join([audio_path, frame_path, str(len(frame_files))]))
                
            
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
