# SIU KING WAI SM4701 Deepstory
import re
import os
import numpy as np
import scipy
import modules.sda as sda
import glob
import torch

from io import BytesIO
from more_itertools import intersperse
from util import normalize_text, separate, save_video
from voice import Voice
from generate import Generator
from animator import ImageAnimator
from modules.dctts import get_silence, hp


class Deepstory:
    def __init__(self):
        # remove previously created video
        if self.is_animated:
            os.remove('export/animated.mp4')
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
        self.wav = None
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

        speaker_dict = {}
        for i, sentence_dict in enumerate(sentence_dicts):
            if sentence_dict['speaker'] not in speaker_dict:
                speaker_dict[sentence_dict['speaker']] = []
            speaker_dict[sentence_dict['speaker']].append(i)
        self.speaker_dict = speaker_dict
        self.sentence_dicts = sentence_dicts

    def modify_speaker(self, speaker_list):
        for i, speaker in enumerate(speaker_list):
            self.sentence_dicts[i]['speaker'] = speaker

    def synthesize_wavs(self):
        for speaker, sentence_ids in self.speaker_dict.items():
            with Voice(speaker) as voice:
                for i in sentence_ids:
                    self.sentence_dicts[i]['wav'] = voice.synthesize(self.sentence_dicts[i]['text'])

    @property
    def is_synthesized(self):
        return 'wav' in self.sentence_dicts[0] if self.sentence_dicts else False

    def combine_wavs(self):
        """Concat wavs of same speaker, so that video of speaker can be made easily"""
        wavs_dicts = []
        wavs_dict = {}
        last_speaker = ''
        for i, sentence_dict in enumerate(self.sentence_dicts):
            wav = sentence_dict['wav']
            # Add silence between lines
            if sentence_dict['begin']:
                wav = np.pad(wav, (get_silence(0.5), 0), 'constant')  # Every line has 0.5s silence

            if i != 0 and last_speaker != sentence_dict['speaker']:
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
        self.wav = np.concatenate([wavs_dict['wav'] for wavs_dict in wavs_dicts], axis=None)
        self.wavs_dicts = wavs_dicts
        # scipy.io.wavfile.write('export/combined.wav', hp.sr, self.wav)

    @property
    def is_combined(self):
        return False if self.wav is None else True

    def stream(self, sentence_id=0, combined=False):
        wav = self.wav if combined else self.sentence_dicts[sentence_id]['wav']
        with BytesIO() as f:
            scipy.io.wavfile.write(f, hp.sr, wav)
            return f.getvalue()

    def wav_to_vid(self):
        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        for i, wavs_dict in enumerate(self.wavs_dicts):
            self.wavs_dicts[i]['base'] = va('data/sda/image.bmp', wavs_dict['wav'], fs=hp.sr)
        del va
        torch.cuda.empty_cache()

    @property
    def is_base(self):
        return 'base' in self.wavs_dicts[0] if self.wavs_dicts else False

    def animate_image(self, image_dict):
        with ImageAnimator() as animator:
            for i, wavs_dict in enumerate(self.wavs_dicts):
                self.wavs_dicts[i]['animated'] = animator.animate_image(
                    f'data/images/{image_dict[wavs_dict["speaker"]]}', wavs_dict['base'])
        save_video(
            np.concatenate([wavs_dict['base'] for wavs_dict in self.wavs_dicts]),
            self.wav, 'export/base.mp4', hp.sr)
        save_video(
            np.concatenate([wavs_dict['animated'] for wavs_dict in self.wavs_dicts]),
            self.wav, 'export/animated.mp4', hp.sr)

    @property
    def is_animated(self):
        return os.path.isfile('export/animated.mp4')
