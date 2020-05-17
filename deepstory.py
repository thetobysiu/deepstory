# SIU KING WAI SM4701 Deepstory
import re
import os
import numpy as np
import scipy
import modules.sda as sda
import glob
import torch
import ffmpeg

from io import BytesIO
from more_itertools import intersperse
from util import normalize_text, separate
from voice import Voice
from generate import Generator
from animator import ImageAnimator
from modules.dctts import get_silence, hp


class Deepstory:
    def __init__(self):
        # # remove previously created video
        # if self.is_animated:
        #     os.remove('export/animated.mp4')
        #     for path in glob.glob('temp/animated/*'):
        #         os.remove(path)
        # if self.is_combined:
        #     os.remove('export/combined.wav')
        # if self.is_base:
        #     for path in glob.glob('temp/base/*'):
        #         os.remove(path)

        self.text = 'Geralt|I hate portals. A round of Gwent maybe?'
        self.generated_text = 'Geralt wants to'
        self.speaker_dict = {}
        self.image_dict = {
            os.path.basename(os.path.dirname(path)): sorted(
                [os.path.basename(file) for file in glob.glob(f'{path}/*.*')])
            for path in glob.glob('data/images/*/')
        }
        self.sentence_dicts = []
        self.wavs_dicts = []
        self.gpt2 = False
        self.gpt2_list = [os.path.split(os.path.split(path)[0])[-1] for path in glob.glob('data/gpt2/*/')]
        self.model_list = [os.path.split(os.path.split(path)[0])[-1] for path in glob.glob('data/dctts/*/')]

    def load_gpt2(self, model_name):
        if self.gpt2:
            del self.gpt2
            torch.cuda.empty_cache()
        self.gpt2 = Generator(model_name)

    @property
    def current_gpt2(self):
        return self.gpt2.model_name if self.gpt2 else False

    def generate_gpt2(self, text, max_length, top_p, top_k, temperature, do_sample):
        self.generated_text = self.gpt2.generate(text, max_length, top_p, top_k, temperature, do_sample)

    def parse_text(self, text, default_speaker, separate_comma=False,
                   n_gram=2, separate_sentence=False, parse_speaker=True, normalize=True):
        """
        Parse the input text into suitable data structure
        :param n_gram: concat sentences of this max length in a line
        :param text: source
        :param default_speaker: the default speaker if no speaker in specified
        :param separate_comma: split by comma
        :param separate_sentence: split sentence if multiple clauses exist
        :param parse_speaker: bool for turn on/off parse speaker
        :param normalize: to convert common punctuation besides comma to comma
        """

        lines = re.split(r'\r\n|\n\r|\r|\n', text)

        line_speaker_dict = {}
        # TODO: allow speakers not in model_list and later are forced to be replaced
        if parse_speaker:
            # re.match(r'^.*(?=:)', text)
            for i, line in enumerate(lines):
                if re.search(r':|\|', line):
                    # ?: non capture group of : and |
                    speaker, line = re.split(r'\s*(?::|\|)\s*', line, 1)
                    # add entry only if the voice model exist in the folder,
                    # the unrecognized one will be changed to default in later code
                    if speaker in self.model_list:
                        line_speaker_dict[i] = speaker
                    lines[i] = line

        if normalize:
            lines = [normalize_text(line) for line in lines]

        # separate by spacy sentencizer
        lines = [separate(line, n_gram, comma=separate_comma) for line in lines]

        sentence_dicts = []
        for i, line in enumerate(lines):
            for j, sent in enumerate(line):
                if sentence_dicts:
                    if sent[0].is_punct and not any(sent[0].text == punct for punct in ['“', '‘']):
                        sentence_dicts[-1]['punct'] = sentence_dicts[-1]['punct'] + sent.text
                        continue
                sentence_dict = {
                    'text': sent.text,
                    'begin': True if j == 0 else False,
                    'punct': '',
                    'speaker': line_speaker_dict.get(i, self.model_list[default_speaker])
                }

                while not sentence_dict['text'][-1].isalpha():
                    sentence_dict['punct'] = sentence_dict['punct'] + sentence_dict['text'][-1]
                    sentence_dict['text'] = sentence_dict['text'][:-1]
                # Reverse the punctuation order since I add it based on the last item
                sentence_dict['punct'] = sentence_dict['punct'][::-1]
                sentence_dict['text'] = sentence_dict['text'] + sentence_dict['punct']
                sentence_dicts.append(sentence_dict)

        self.sentence_dicts = sentence_dicts
        self.update_speaker_dict()

    def update_speaker_dict(self):
        speaker_dict = {}
        for i, sentence_dict in enumerate(self.sentence_dicts):
            if sentence_dict['speaker'] not in speaker_dict:
                speaker_dict[sentence_dict['speaker']] = []
            speaker_dict[sentence_dict['speaker']].append(i)
        self.speaker_dict = speaker_dict

    def modify_speaker(self, speaker_list):
        for i, speaker in enumerate(speaker_list):
            self.sentence_dicts[i]['speaker'] = speaker
        self.update_speaker_dict()

    def synthesize_wavs(self):
        # clear model from vram to revent out of memory error
        if self.current_gpt2:
            del self.gpt2
            self.gpt2 = None
            torch.cuda.empty_cache()
        for speaker, sentence_ids in self.speaker_dict.items():
            with Voice(speaker) as voice:
                for i in sentence_ids:
                    self.sentence_dicts[i]['wav'] = voice.synthesize(self.sentence_dicts[i]['text'])

    @property
    def is_synthesized(self):
        return 'wav' in self.sentence_dicts[0] if self.sentence_dicts else False

    def combine_wavs(self, cut_size=800000):
        """Concat wavs of same speaker, so that video of speaker can be made easily"""
        # adjust the cut_size if you have vram issue
        wavs_dicts = []
        wavs_dict = {}
        last_speaker = ''
        for i, sentence_dict in enumerate(self.sentence_dicts):
            wav = sentence_dict['wav']
            # Add silence between lines
            if sentence_dict['begin']:
                wav = np.pad(wav, (get_silence(0.5), 0), 'constant')  # Every line has 0.5s silence

            if i != 0 and (last_speaker != sentence_dict['speaker'] or sum(len(wav) for wav in wavs_dict['wav']) > cut_size):
                wavs_dict['speaker'] = last_speaker
                # Add silence between each sentence within a line, default 0.15s
                wavs_dict['wav'] = np.concatenate(
                    [*intersperse(np.zeros(get_silence(0.15), dtype=np.int16), wavs_dict['wav'])], axis=None)
                # pad silence at the end
                wavs_dict['wav'] = np.pad(wavs_dict['wav'], (0, get_silence(0.5)), 'constant')
                wavs_dicts.append(wavs_dict)
                wavs_dict = {}

            if 'wav' not in wavs_dict:
                wavs_dict['wav'] = [wav]
            else:
                wavs_dict['wav'].append(wav)
            last_speaker = sentence_dict['speaker']

        if wavs_dict:
            wavs_dict['speaker'] = last_speaker
            # Add silence between each sentence within a line, default 0.15s
            wavs_dict['wav'] = np.concatenate(
                [*intersperse(np.zeros(get_silence(0.15), dtype=np.int16), wavs_dict['wav'])], axis=None)
            # pad silence at the end
            wavs_dict['wav'] = np.pad(wavs_dict['wav'], (0, get_silence(0.5)), 'constant')
            wavs_dicts.append(wavs_dict)
        # TODO: add silence according to punctuation
        scipy.io.wavfile.write('export/combined.wav', hp.sr,
                               np.concatenate([wavs_dict['wav'] for wavs_dict in wavs_dicts], axis=None))
        self.wavs_dicts = wavs_dicts

    @property
    def is_combined(self):
        return os.path.isfile('export/combined.wav')

    def stream(self, sentence_id=0):
        with BytesIO() as f:
            scipy.io.wavfile.write(f, hp.sr, self.sentence_dicts[sentence_id]['wav'])
            return f.getvalue()

    def wav_to_vid(self):
        """Create Base video for First Order Motion Model"""
        if not os.path.isdir(f'temp'):
            os.mkdir(f'temp')
        if os.path.isdir(f'temp/base'):
            for path in glob.glob('temp/base/*'):
                os.remove(path)
        else:
            os.mkdir(f'temp/base')

        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        for i, wavs_dict in enumerate(self.wavs_dicts):
            np.save(f'temp/base/{i}|{wavs_dict["speaker"]}.npy', va('data/sda/image.bmp', wavs_dict['wav'], fs=hp.sr))
        del va
        del self.wavs_dicts
        self.wavs_dicts = []
        torch.cuda.empty_cache()

    @property
    def is_base(self):
        try:
            return bool(os.listdir('temp/base'))
        except FileNotFoundError:
            return False

    def animate_image(self, image_dict):
        if os.path.isdir(f'temp/animated'):
            for path in glob.glob('temp/animated/*'):
                os.remove(path)
        else:
            os.mkdir(f'temp/animated')

        with ImageAnimator() as animator:
            for i, base_path in enumerate(sorted(
                    glob.glob('temp/base/*'),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("|")[0]))):
                speaker = os.path.splitext(os.path.basename(base_path))[0].split("|")[1]
                animator.animate_image(
                    f'data/images/{image_dict[speaker]}',
                    np.load(base_path),
                    f'temp/animated/{i}.mp4'
                )

        audio = ffmpeg.input('export/combined.wav').audio
        videos = [ffmpeg.input(clip).video for clip in sorted(
            glob.glob('temp/animated/*'), key=lambda x: int(os.path.basename(x)[:-4]))]
        ffmpeg.concat(*videos).output('export/combined.mp4', loglevel="panic").overwrite_output().run()
        video = ffmpeg.input('export/combined.mp4').video
        ffmpeg.output(
            video, audio, 'export/animated.mp4', loglevel="panic", vcodec="copy", ar=hp.sr, **{'b:a': '128k'}
        ).overwrite_output().run()

    @property
    def is_animated(self):
        return os.path.isfile('export/animated.mp4')
