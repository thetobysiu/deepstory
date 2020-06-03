# SIU KING WAI SM4701 Deepstory
from flask import Flask, render_template, request, make_response, send_from_directory
from deepstory import Deepstory
app = Flask(__name__)
deepstory = Deepstory()
print('Deepstory instance created')


def send_message(message, status=200):
    response = make_response(message, status)
    response.mimetype = "text/plain"
    return response


@app.route('/')
def index():
    return render_template('index.html', model_list=deepstory.model_list, gpt2_list=deepstory.gpt2_list,
                           image_dict=deepstory.image_dict, speaker_map=deepstory.speaker_map_dict)


@app.route('/map')
def map_page():
    return render_template('map.html', model_list=deepstory.model_list, speaker_map=deepstory.speaker_map_dict)


@app.route('/deepstory.js')
def deepstoryjs():
    return render_template('deepstory.js')


@app.route('/status')
def status():
    return render_template('status.html', synthsized=deepstory.is_synthesized,
                           combined=deepstory.is_processed, base=deepstory.is_base, animated=deepstory.is_animated)


@app.route('/gpt2')
def gpt2():
    return render_template('gpt2.html', gpt2=deepstory.current_gpt2, generated_text=deepstory.generated_text)


@app.route('/gen_sents')
def gen_sents():
    return render_template('gen_sentences.html', sentences=deepstory.generated_sentences)


@app.route('/sentences')
def sentences():
    return render_template('sentences.html',
                           sentences=deepstory.sentence_dicts, model_list=deepstory.model_list,
                           speaker_map=deepstory.speaker_map_dict)


@app.route('/load_text')
def load_text():
    model = request.args.get('model')
    lines_no = int(request.args.get('lines'))
    try:
        deepstory.load_text(model, lines_no)
        return send_message(f'{lines_no} in {model} loaded.')
    except FileNotFoundError:
        return send_message(f'Text file not found.', 403)


@app.route('/load_gpt2', methods=['GET'])
def load_gpt2():
    model = request.args.get('model')
    if deepstory.gpt2:
        if deepstory.current_gpt2 == model:
            return send_message(f'{model} is already loaded.', 403)
    deepstory.load_gpt2(model)
    return send_message(f'{model} loaded.')


@app.route('/generate_text', methods=['POST'])
def generate_text():
    if deepstory.gpt2:
        text = request.form.get('text')
        predict_length = int(request.form.get('predict_length'))
        top_p = float(request.form.get('top_p'))
        top_k = int(request.form.get('top_k'))
        temperature = float(request.form.get('temperature'))
        do_sample = bool(request.form.get('do_sample'))
        deepstory.generate_text_gpt2(text, predict_length, top_p, top_k, temperature, do_sample)
        return send_message(f'Generated.')
    else:
        return send_message(f'Please load a GPT2 model first.', 403)


@app.route('/generate_sents', methods=['POST'])
def generate_sents():
    if deepstory.gpt2:
        text = request.form.get('text')
        predict_length = int(request.form.get('predict_length'))
        top_p = float(request.form.get('top_p'))
        top_k = int(request.form.get('top_k'))
        temperature = float(request.form.get('temperature'))
        do_sample = bool(request.form.get('do_sample'))
        batches = int(request.form.get('batches'))
        max_sentences = int(request.form.get('max_sentences'))
        deepstory.generate_sents_gpt2(
            text, predict_length, top_p, top_k, temperature, do_sample, batches, max_sentences)
        return send_message(f'Generated.')
    else:
        return send_message(f'Please load a GPT2 model first.', 403)


@app.route('/add_sent', methods=['GET'])
def add_sent():
    sent_id = int(request.args.get('id'))
    deepstory.add_sent(sent_id)
    return send_message(f'Sentence {sent_id} added.')


@app.route('/load_sentence', methods=['POST'])
def load_sentence():
    text = request.form.get('text')
    speaker = request.form.get('speaker')
    if text:
        is_comma = bool(request.form.get('isComma'))
        is_chopped = bool(request.form.get('isChopped'))
        is_speaker = bool(request.form.get('isSpeaker'))
        force = bool(request.form.get('force'))
        n = int(request.form.get('n'))
        deepstory.parse_text(text,
                             n_gram=n,
                             default_speaker=speaker,
                             separate_comma=is_comma,
                             separate_sentence=is_chopped,
                             parse_speaker=is_speaker,
                             force_parse=force)
        return send_message(f'Sentences loaded.')
    else:
        return send_message('Please enter text.', 403)


@app.route('/animate', methods=['GET', 'POST'])
def animate():
    if request.method == 'POST':
        deepstory.animate_image(request.form)
        return send_message(f'Images animated.')
    elif request.method == 'GET':
        return render_template('animate.html',
                               image_dict=deepstory.image_dict,
                               loaded_speakers=deepstory.get_base_speakers())


@app.route('/modify', methods=['POST'])
def modify():
    deepstory.modify_speaker(request.json)
    return send_message(f'Speaker updated.')


@app.route('/clear')
def clear():
    deepstory.clear_cache()
    return send_message(f'Cache cleared.')


@app.route('/update_map', methods=['POST'])
def update_map():
    deepstory.update_mapping(request.form)
    return send_message(f'Mapping updated.')


@app.route('/synthesize')
def synthesize():
    if deepstory.sentence_dicts:
        try:
            deepstory.synthesize_wavs()
            return send_message(f'Sentences synthesized.')
        except FileNotFoundError:
            return send_message("One of the model doesn't exist, please modify to something else.", 403)
    else:
        return send_message('Please enter text.', 403)


@app.route('/combine')
def combine():
    try:
        deepstory.process_wavs()
        return send_message(f'Clip created.')
    except (KeyError, ValueError):
        return send_message('No audio is synthesized to be combined', 403)
    # except:
    #     return send_message('Unknown Error.', 403)


@app.route('/create_base')
def create_base():
    if deepstory.is_processed:
        deepstory.wav_to_vid()
        return send_message(f'Base video created.')
    else:
        return send_message('No audio is synthesized to be processed', 403)


@app.route("/wav/<int:sentence_id>")
def stream(sentence_id):
    response = make_response(deepstory.stream(sentence_id), 200)
    response.mimetype = "audio/x-wav"
    return response


@app.route('/image/<path:filename>')
def image_viewer(filename):
    return send_from_directory(f'data/images/', filename)


@app.route('/video')
def video():
    return render_template('video.html', animated=deepstory.is_animated)


@app.route('/get_video')
def video_viewer():
    return send_from_directory(f'export', 'animated.mp4')


if __name__ == '__main__':
    app.run()
