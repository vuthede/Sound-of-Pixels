import librosa
import torch
import av
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import random
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#import matplotlib.pyplot as plt
#import librosa.display
#import moviepy.editor as mpy

class LRS2(torch.utils.data.Dataset):
    def __init__(self, root_dir, split = "train", duration = 2, sampling_rate = 16000, fps = 25, hop_rate = 0.01, n_fft = 512, n_mels = 80, num_negative = 5):
        super(LRS2, self).__init__()

        # Load file txt contains training information data
        self.filelist = []
        path = os.path.join(root_dir, "saved_" + split + ".npy")
        if os.path.exists(path):
            self.filelist = np.load(path)
        else:
            with open(os.path.join(root_dir, split + ".txt"), "r") as reader:
                for line in reader:
                    video_name = line.rstrip("\r\n")
                    video_path = os.path.join(root_dir, "main", video_name + ".mp4")
                    container = av.open(video_path)
                    length = float(container.streams.video[0].frames / container.streams.video[0].rate)
                    if length > duration:
                        self.filelist.append(video_path)
                np.save(path, np.array(self.filelist))

        # Other params
        self.FPS = fps
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_rate = hop_rate
        self.sampling_rate = sampling_rate
        self.hop_length = int(hop_rate * self.sampling_rate)
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        self.duration = duration
        self.num_negative = num_negative
        self.split = "split"

    def __len__(self):
        return len(self.filelist)

    def write_video(self, path, frames, waveform):
        def makeframe(t):
            t = t * self.sampling_rate
            if type(t) is np.ndarray:
                t = t.astype(np.int32)
            else:
                t = int(t)
            return [waveform[0, t]] #, waveform[0, t]]

        clip = mpy.ImageSequenceClip(frames, fps=25)
        audio = mpy.AudioClip(makeframe, duration = len(frames) / 25.)
        clip.audio = audio
        clip.write_videofile(path)

        S = librosa.feature.melspectrogram(waveform[0, :], sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(S_DB, sr=self.sampling_rate, hop_length=self.hop_length, ax = ax, x_axis='time', y_axis='mel')
        path = path.replace(".mp4","_spec.jpeg")
        fig.savefig(path)



    def load_frames_and_waveform(self, index):
        container = av.open(self.filelist[index])
        frames = []
        audios = []

        sampling_rate = container.streams.audio[0].rate
        fps = container.streams.video[0].rate
        assert fps == self.FPS, "fps of %s is %s. expected fps is %d" % (fps, self.filelist[index], self.FPS)
        assert sampling_rate == self.sampling_rate, "sampling rate of %s is %s. expected sampling rate is %d" % (fps, self.filelist[index], self.sampling_rate)

        for data in container.decode():
            if type(data) is av.audio.frame.AudioFrame:
                audios.append(data)
            else:
                frames.append(data)

        waveform = np.concatenate([audio.to_ndarray()
                                     for audio in audios], axis = 1)
        if waveform.shape[0] != 1:
            waveform = waveform.mean(axis = 0, keepdims = True)
        return frames, waveform, sampling_rate, fps

    def project_waveform(self, waveform, sampling_rate, start_second, end_second):
        start_index = int(sampling_rate * start_second)
        end_index = int(sampling_rate * end_second)
        subwaveform = waveform[:, start_index:end_index]
        expected_length = int(self.duration * sampling_rate)
        if subwaveform.shape[1] <  expected_length:
            subwaveform = np.pad(subwaveform, ((0, 0), (0, expected_length - subwaveform.shape[1])))
        return subwaveform[:, :expected_length]

    def project_frames(self, frames, fps, start_second, end_second):
        rgb_frames = []
        for frame in frames:
            t = frame.pts * frame.time_base
            if (start_second <= t) and (t <= end_second):
                img = frame.to_rgb().to_ndarray().astype(np.float) / 255
                img = (img -self. mean) / self.std
                rgb_frames.append(img) #frame.to_rgb().to_ndarray())


        expected_length = int(self.duration * fps)
        if len(rgb_frames) < expected_length:
            last_frame = rgb_frames[-1]
            pad = expected_length - len(rgb_frames)
            for i in range(pad):
                rgb_frames.append(last_frame)

        rgb_frames = np.stack(rgb_frames[:expected_length])
        rgb_frames = torch.as_tensor(rgb_frames) # T x H x W x C
        rgb_frames = rgb_frames.to(torch.float)
        return rgb_frames.permute(3, 0, 1, 2) # C x T x H x W


    def load_video(self, index):
        frames, waveform, sampling_rate, fps = self.load_frames_and_waveform(index)
        start_index = 0
        start_second = frames[start_index].pts * frames[start_index].time_base
        end_second = start_second + self.duration
        positive_waveform = self.project_waveform(waveform, sampling_rate, start_second, end_second)
        rgb_frames = self.project_frames(frames, fps, start_second, end_second)

        hop_length = int(sampling_rate * self.hop_rate)
        mel_spec = librosa.feature.melspectrogram(positive_waveform[0, :-1], sr = sampling_rate, n_fft = self.n_fft, hop_length=hop_length, n_mels = self.n_mels)
        mel_spec = torch.as_tensor(mel_spec)
        mel_spec = mel_spec.permute(1, 0)
        mel_spec = mel_spec.unsqueeze(0)
        return rgb_frames, mel_spec, positive_waveform

    def sampling_different_videos(self, positive_index):
        video_name = "filename_%d_%s" % (positive_index, os.path.basename(self.filelist[positive_index]))
        video_name = video_name.split(".")[0]
        frames, positive_spec, positive_waveform = self.load_video(positive_index)

        #np_frames = []
        #for i in range(frames.size(1)):
        #    img = frames[:, i, :, :].numpy().transpose(1, 2, 0)
        #    img = np.clip((img * self.std + self.mean) * 255., 0, 255).astype(np.uint8)
        #    np_frames.append(img)

        negative_specs = []
        selected_indicies = [positive_index]
        #self.write_video("outputs/%s_0_positive.mp4" % video_name, np_frames, positive_waveform)
        for i in range(self.num_negative):
            while True:
                negative_index = random.randint(0, len(self.filelist) - 1)
                if negative_index not in selected_indicies:
                    break
            #print("load", negative_index)
            _, negative_spec, negative_waveform = self.load_video(negative_index)
            #self.write_video("outputs/%s_%d_negative.mp4" % (video_name, i + 1), np_frames, negative_waveform)

            negative_specs.append(negative_spec)
            selected_indicies.append(negative_index)

        return frames, positive_spec, torch.stack(negative_specs)

    def sampling_within_video(self, index):
        raw_frames, raw_waveform, sampling_rate, fps = self.load_frames_and_waveform(index)
        specs = []
        frames = []
        length = raw_waveform.shape[1] / sampling_rate
        selected_second = []
        hop_length = int(sampling_rate * self.hop_rate)

        mags = []
        phases = []
        waves = []

        for i in range(self.num_negative + 1):
            stop = False
            while not stop:
                start_second = random.uniform(0, length - self.duration)
                for second in selected_second:
                    if np.abs(start_second - second) <= 0.5:
                        stop = False
                        break
                stop = True
                #if start_second not in selected_second:
                #    break

            end_second = start_second + self.duration
            selected_second.append(start_second)
            frame = self.project_frames(raw_frames, fps, start_second, end_second)
            waveform = self.project_waveform(raw_waveform, sampling_rate, start_second, end_second)
            #print(start_second, end_second, waveform.shape)
            # mel_spec = librosa.feature.melspectrogram(waveform[0, :-1], sr = sampling_rate, n_fft = self.n_fft, hop_length=hop_length, n_mels = self.n_mels)
            # mel_spec = torch.as_tensor(mel_spec)
            # mel_spec = mel_spec.permute(1, 0)
            # mel_spec = mel_spec.unsqueeze(0)
            # mag, phase = librosa.stft(waveform, n_fft=self.n_fft, hop_length=hop_length)


            frames.append(frame)
            waves.append(waveform)
            # specs.append(mel_spec)
            # mags.append(mag)
            # phases.append(phase)

        # return frames[0], specs[0], torch.stack(specs[1:])
        return frames[0], waves[0][0]
    
    def getvideoname(self, index):
        return self.filelist[index]

    def getitem(self, index):
        return self.sampling_within_video(index)

if __name__ == "__main__":
    pass
    # import wave, struct

    # dataset = LRS2("/media/Data_SSD/ttdat/LRS2", "train", duration = 1, num_negative = 4)
    # for i in range(5):
    #     a, b, c = dataset[i]
    #     print(a.shape, b.shape, c.shape)

