import os
import random
from .base import BaseDataset
import glob
import numpy as np

class MUSICMixDataset(BaseDataset):
    # Used for silence video
    # SCENCE_IMAGE_DIR = "/home/ubuntu/MyHelperModule/downloaddata/download_data_for_sound_of_pixel_paper/image_sence"
    def __init__(self, scence_image_dir, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.list_scence_image =  glob.glob(scence_image_dir + "/*.jpg") # For synthesize silence sample
        print(f'Len of list scence image: {len(self.list_scence_image)}')
    def __random_silence_video(self):
        # 10% will be silience video
        a = np.random.randint(0,10)
        if a>=-1:
            return False
        return True

    def __random_scence_image(self):
        index = np.random.randint(0,len(self.list_scence_image)-1)
        return self.list_scence_image[index]

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]

        # select frames
       # idx_margin = max(
        #    int(self.fps * 4), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                #print(f'idx_margin: {idx_margin}. countframes:{count_framesN}')
                #center_frameN = random.randint(
                 #   idx_margin+1, int(count_framesN)-idx_margin)
                center_frameN = random.randint(10, 39)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN
        
        # Random 30 % using silent audio to regulalize model
        using_silence = self.__random_silence_video()
        if using_silence and len(infos) >1:
            print("Using silence image and video!!!!!!!!!!!!!!!!!!!!!!!!")
            path_frames[1]= []
            for i in range(self.num_frames):
                path_frames[1].append(self.__random_scence_image())
            path_audios[1] = "FakeSilenceFile.silent" # it can be any string as long as it ends by "silent"
        #else:
            #print("Using normal image and video")

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
