import os
import random
from .base import BaseDataset
from .baselips import LRS2
import glob
import numpy as np
import time
import librosa
import torch


class LIPSMIXDATASET(LRS2):
    # Used for silence video
    # SCENCE_IMAGE_DIR = "/home/ubuntu/MyHelperModule/downloaddata/download_data_for_sound_of_pixel_paper/image_sence"
    def __init__(self,  root_dir, split = "train", duration=3,**kwargs):
        super(LIPSMIXDATASET, self).__init__(root_dir=root_dir, split=split,duration=duration,**kwargs)

    def _stft(self, audio):
        audio = audio[:self.audLen-1*self.stft_hop]
        spec = librosa.stft(
            audio, sr=self.sampling_rate, n_fft = self.n_fft, hop_length=self.hop_length)
        #print("Spec shape: ", spec.shape)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)
        
        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        # audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)



    def __getitem__(self, index):
        t1 =  time.time()
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        mags = [None for n in range(N)]


        # the first sample
        infos[0] = [self.getvideoname(index), index]  # 123213213213/00024.mp4 , 4

        # Get other sample
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, self.__len__() -1)
            info[n] = [self.getvideoname(indexN), indexN]


        # Mix N samples
        try:
            for n, infoN in enumerate(infos):
                data = self.getitem(infos[n][1])
                frames[n] = data[0]
                audios[n] = data[1]
            
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
       

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        #print("1 item")
        if index%100==0:
            print("Time to load and preapre a sample:", time.time()-t1)
        
        return ret_dict
