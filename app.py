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
                           image_dict=deepstory.image_dict)


@app.route('/deepstory.js')
def deepstoryjs():
    return render_template('deepstory.js')


@app.route('/status')
def status():
    return render_template('status.html', gpt2=deepstory.current_gpt2, synthsized=deepstory.is_synthesized,
                           combined=deepstory.is_combined, base=deepstory.is_base, animated=deepstory.is_animated)


@app.route('/gpt2')
def gpt2():
    return render_template('gpt2.html', generated_text=deepstory.generated_text)


@app.route('/sentences')
def sentences():
    return render_template('sentences.html',
                           sentences=deepstory.sentence_dicts, model_list=deepstory.model_list)


@app.route('/load_gpt2', methods=['GET'])
def load_gpt2():
    model = request.args.get('model')
    if deepstory.gpt2:
        if deepstory.current_gpt2 == model:
            return send_message(f'{model} is already loaded.', 403)
    deepstory.load_gpt2(model)
    return send_message(f'{model} loaded.')


@app.route('/generate', methods=['POST'])
def generate():
    if deepstory.gpt2:
        text = request.form.get('text')
        max_length = int(request.form.get('max_length'))
        top_p = float(request.form.get('top_p'))
        top_k = int(request.form.get('top_k'))
        temperature = float(request.form.get('temperature'))
        do_sample = bool(request.form.get('do_sample'))
        deepstory.generate_gpt2(text, max_length, top_p, top_k, temperature, do_sample)
        return send_message(f'Generated.')
    else:
        return send_message(f'Please load a GPT2 model first.', 403)


@app.route('/load_sentence', methods=['POST'])
def load_sentence():
    text = request.form.get('text')
    speaker = int(request.form.get('speaker'))
    if text:
        is_comma = bool(request.form.get('isComma'))
        is_chopped = bool(request.form.get('isChopped'))
        is_speaker = bool(request.form.get('isSpeaker'))
        n = int(request.form.get('n'))
        deepstory.parse_text(text,
                             n_gram=n,
                             default_speaker=speaker,
                             separate_comma=is_comma,
                             separate_sentence=is_chopped,
                             parse_speaker=is_speaker)
        return send_message(f'Sentences loaded.')
    else:
        return send_message('Please enter text.', 403)


@app.route('/animate', methods=['GET'])
def animate():
    if deepstory.is_base:
        deepstory.animate_image(request.args)
        return send_message(f'Images animated.')
    else:
        return send_message('Please create base video first.', 403)


@app.route('/modify', methods=['POST'])
def modify():
    deepstory.modify_speaker(request.json)
    return send_message(f'Speaker updated.')


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
        deepstory.combine_wavs()
        return send_message(f'Clip created.')
    except (KeyError, ValueError):
        return send_message('No audio is synthesized to be combined', 403)
    # except:
    #     return send_message('Unknown Error.', 403)


@app.route('/create_base')
def create_base():
    try:
        if deepstory.wavs_dicts:
            deepstory.wav_to_vid()
            return send_message(f'Base video created.')
        else:
            raise KeyError
    except KeyError:
        return send_message('No audio is synthesized to be combined', 403)


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
    return send_from_directory(f'export', 'animated.mp4')


if __name__ == '__main__':
    app.run()
