# SIU KING WAI SM4701 Deepstory
import re
import os
import numpy as np
import scipy
import modules.sda as sda
import glob
import torch
import ffmpeg
import random

from io import BytesIO
from util import separate, fix_text, trim_text, split_audio_to_list, get_duration
from voice import Voice
from generate import Generator
from animator import ImageAnimator
from modules.dctts import hp


class Deepstory:
    def __init__(self):
        # remove previously created video
        # self.clear_cache()
        # self.text = 'Geralt|I hate portals. A round of Gwent maybe?'
        self.generated_text = ''
        self.generated_sentences = []
        self.speaker_dict = {}
        self.speaker_map_dict = {}
        self.image_dict = {
            os.path.basename(os.path.dirname(path)): sorted(
                [os.path.basename(file) for file in glob.glob(f'{path}/*.*')])
            for path in glob.glob('data/images/*/')
        }
        self.sentence_dicts = []
        self.gpt2 = False
        self.gpt2_list = [os.path.split(os.path.split(path)[0])[-1] for path in glob.glob('data/gpt2/*/')]
        self.speaker_list = []
        self.model_list = [os.path.split(os.path.split(path)[0])[-1] for path in glob.glob('data/dctts/*/')]

    def load_gpt2(self, model_name):
        if self.gpt2:
            del self.gpt2
            torch.cuda.empty_cache()
        self.gpt2 = Generator(model_name)
        self.generated_text = self.gpt2.default
        self.generated_sentences = []

    def load_text(self, model_name, lines_no):
        with open(f'data/gpt2/{model_name}/text.txt', 'r') as f:
            lines = f.readlines()
        start_index = random.randint(0, len(lines) - 1 - lines_no)
        text = ''.join(lines[start_index:start_index+lines_no])
        if text[-1] == '\n':
            text = text[:-1]
        self.generated_text = text

    @property
    def current_gpt2(self):
        return self.gpt2.model_name if self.gpt2 else False

    def generate_text_gpt2(self, text, predict_length, top_p, top_k, temperature, do_sample):
        self.generated_sentences = []
        script = self.current_gpt2 == 'Waiting for Godot'
        result = trim_text(self.gpt2.generate(text, predict_length, top_p, top_k, temperature, do_sample)[0],
                           script=script)
        self.generated_text = text + result

    def generate_sents_gpt2(self, text, predict_length, top_p, top_k, temperature, do_sample, batches, max_sentences):
        self.generated_text = text
        script = self.current_gpt2 == 'Waiting for Godot'
        sents = self.gpt2.generate(text, predict_length, top_p, top_k, temperature, do_sample, num=batches)
        self.generated_sentences = [trim_text(sent, max_sentences, script=script) for sent in sents]

    def add_sent(self, sent_id):
        self.generated_text += self.generated_sentences[sent_id]
        self.generated_sentences = []

    def parse_text(self, text, default_speaker, force_parse=False, separate_comma=False,
                   n_gram=2, separate_sentence=False, parse_speaker=True, normalize=True):
        """
        Parse the input text into suitable data structure
        :param force_parse: forced to replace all speaker that are not in model list as the default speaker
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
        self.speaker_list = []
        self.speaker_map_dict = {}
        if parse_speaker:
            # re.match(r'^.*(?=:)', text)
            for i, line in enumerate(lines):
                if re.search(r':|\|', line):
                    # ?: non capture group of : and |
                    speaker, line = re.split(r'\s*(?::|\|)\s*', line, 1)
                    # add entry only if the voice model exist in the folder,
                    # the unrecognized one will need to mapped so as to be able to be synthesized
                    if force_parse:
                        if speaker in self.model_list:
                            line_speaker_dict[i] = speaker
                    else:
                        if speaker not in self.speaker_list:
                            self.speaker_list.append(speaker)
                        line_speaker_dict[i] = speaker
                    lines[i] = line

            for i, speaker in enumerate(self.speaker_list):
                if speaker not in self.model_list:
                    self.speaker_map_dict[speaker] = self.model_list[i % len(self.model_list)]

        # separate by spacy sentencizer
        lines = [separate(fix_text(line), n_gram, comma=separate_comma) for line in lines]

        self.sentence_dicts = []
        for i, line in enumerate(lines):
            for j, sent in enumerate(line):
                if self.sentence_dicts:
                    # might be buggy, forgot why I wrote this at all
                    while sent[0].is_punct and not any(sent[0].text == punct for punct in ['“', '‘']):
                        self.sentence_dicts[-1]['punct'] = self.sentence_dicts[-1]['punct'] + sent.text[0]
                        sent = sent[1:]
                        continue

                sentence_dict = {
                    'text': sent.text,
                    'begin': True if j == 0 else False,
                    'punct': '',
                    'speaker': line_speaker_dict.get(i, default_speaker)
                }

                while not sentence_dict['text'][-1].isalpha():
                    sentence_dict['punct'] = sentence_dict['punct'] + sentence_dict['text'][-1]
                    sentence_dict['text'] = sentence_dict['text'][:-1]
                # Reverse the punctuation order since I add it based on the last item
                sentence_dict['punct'] = sentence_dict['punct'][::-1]
                sentence_dict['text'] = sentence_dict['text'] + sentence_dict['punct']
                self.sentence_dicts.append(sentence_dict)

        self.update_speaker_dict()

    def update_speaker_dict(self):
        self.speaker_dict = {}
        for i, sentence_dict in enumerate(self.sentence_dicts):
            if sentence_dict['speaker'] not in self.speaker_dict:
                self.speaker_dict[sentence_dict['speaker']] = []
            self.speaker_dict[sentence_dict['speaker']].append(i)

    def update_mapping(self, map_dict):
        for speaker, mapped in map_dict.items():
            self.speaker_map_dict[speaker] = mapped

    def modify_speaker(self, speaker_list):
        for i, speaker in enumerate(speaker_list):
            self.sentence_dicts[i]['speaker'] = speaker
        self.update_speaker_dict()

    def synthesize_wavs(self):
        # clear model from vram to prevent out of memory error
        if self.current_gpt2:
            del self.gpt2
            self.gpt2 = None
            torch.cuda.empty_cache()

        speaker_dict_mapped = {}
        for speaker, sentence_ids in self.speaker_dict.items():
            mapped_speaker = self.speaker_map_dict.get(speaker, speaker)
            if mapped_speaker in speaker_dict_mapped:
                speaker_dict_mapped[mapped_speaker].extend(sentence_ids)
            else:
                speaker_dict_mapped[mapped_speaker] = sentence_ids

        for speaker, sentence_ids in speaker_dict_mapped.items():
            with Voice(speaker) as voice:
                for i in sentence_ids:
                    self.sentence_dicts[i]['wav'] = voice.synthesize(self.sentence_dicts[i]['text'])

    @property
    def is_synthesized(self):
        return 'wav' in self.sentence_dicts[0] if self.sentence_dicts else False

    def process_wavs(self):
        """
        Prepare wavs
        1. Add silence at beginning if the sentence is the beginning of a line
        2. Add silence at the end based on punctuation in the sentence
        3. Finely split the audio again so that sentence with comma can be chopped (increasing sda model performance)
        4. Create a cache of the whole combined audio clips
        """
        # can be adjusted as you like, in seconds.
        punctuation_dict = {
            '.': 0.3,
            ',': 0.15,
            '!': 0.3,
            '?': 0.4,
            '"': 0.1,
            '…': 0.6,
            ':': 0.15,
            ';': 0.2,
            '’': 0.05,
            '‘': 0.05,
            '”': 0.05,
            '“': 0.05
        }

        if not os.path.isdir(f'temp'):
            os.mkdir(f'temp')

        if os.path.isdir(f'temp/audio'):
            for path in glob.glob('temp/audio/*'):
                os.remove(path)
        else:
            os.mkdir(f'temp/audio')

        wavs_dicts = []
        for sentence_dict in self.sentence_dicts:
            # Add silence between lines
            pad_begin = get_duration(0.5) if sentence_dict['begin'] else 0
            pad_end = get_duration(sum(float(punctuation_dict.get(punct, 0)) for punct in sentence_dict['punct']))
            wav = np.pad(sentence_dict['wav'], (pad_begin, pad_end), 'constant')
            split_list = split_audio_to_list(wav)
            for i, wav_slice in enumerate(split_list):
                wav_part = wav[slice(*wav_slice)]
                # add some more silence so that the video generated would not be that awkward
                if i == 0:
                    wav_part = np.pad(wav_part, (0, get_duration(0.1)), 'constant')
                elif i == len(split_list) - 1:
                    wav_part = np.pad(wav_part, (get_duration(0.1), 0), 'constant')
                else:
                    # append silence at the beginning of slice
                    wav_part = np.pad(wav_part, (get_duration(0.1), get_duration(0.1)), 'constant')
                wavs_dicts.append({
                    'speaker': sentence_dict['speaker'],
                    'wav': wav_part
                })

        for i, wav_dict in enumerate(wavs_dicts):
            scipy.io.wavfile.write(f'temp/audio/{i:03d}|{wav_dict["speaker"]}.wav', hp.sr, wav_dict['wav'])

        scipy.io.wavfile.write('export/combined.wav', hp.sr,
                               np.concatenate([wavs_dict['wav'] for wavs_dict in wavs_dicts], axis=None))

    @property
    def is_processed(self):
        try:
            return bool(os.listdir('temp/audio'))
        except FileNotFoundError:
            return False

    def stream(self, sentence_id=0):
        with BytesIO() as f:
            scipy.io.wavfile.write(f, hp.sr, self.sentence_dicts[sentence_id]['wav'])
            return f.getvalue()

    @staticmethod
    def wav_to_vid():
        """Create Base video for First Order Motion Model"""
        if os.path.isdir(f'temp/base'):
            for path in glob.glob('temp/base/*'):
                os.remove(path)
        else:
            os.mkdir(f'temp/base')

        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        for audio_path in sorted(
                glob.glob('temp/audio/*.wav'),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("|")[0])
        ):
            np.save(
                f'temp/base/{os.path.splitext(os.path.basename(audio_path))[0]}.npy',
                va('data/sda/image.bmp', audio_path)
            )
        del va
        torch.cuda.empty_cache()

    @property
    def is_base(self):
        try:
            return bool(os.listdir('temp/base'))
        except FileNotFoundError:
            return False

    @staticmethod
    def get_base_speakers():
        return set(
            os.path.splitext(os.path.basename(base_path))[0].split("|")[1]
            for base_path in glob.glob('temp/base/*.npy')
        )

    @staticmethod
    def animate_image(image_dict):
        if os.path.isdir(f'temp/animated'):
            for path in glob.glob('temp/animated/*'):
                os.remove(path)
        else:
            os.mkdir(f'temp/animated')

        with ImageAnimator() as animator:
            for i, base_path in enumerate(sorted(
                    glob.glob('temp/base/*.npy'),
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

    def clear_cache(self):
        # remove previously created video
        if self.is_animated:
            for path in glob.glob('temp/animated/*'):
                os.remove(path)
            os.remove('export/animated.mp4')
            os.remove('export/combined.mp4')
        if self.is_processed:
            for path in glob.glob('temp/audio/*'):
                os.remove(path)
            os.remove('export/combined.wav')
        if self.is_base:
            for path in glob.glob('temp/base/*'):
                os.remove(path)
