from flask import Flask, render_template, request, make_response
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
    return render_template('index.html', model_list=deepstory.model_list)


@app.route('/status')
def status():
    return render_template('status.html')


@app.route('/models')
def models():
    return render_template('models.html', model_list=deepstory.model_list)


@app.route('/sentences')
def sentences():
    return render_template('sentences.html',
                           sentences=deepstory.sentence_dicts, model_list=deepstory.model_list)


# @app.route('/init', methods=['POST'])
# def init_model():
#     speaker = request.form.get('speaker')
#     if speaker:
#         if deepstory.models.get(speaker):
#             return send_message(f'{speaker} is already initialized.', 403)
#         else:
#             deepstory.init_model(speaker, f'data/dctts/{speaker}')
#             return send_message(f'{speaker} initialized.')
#     else:
#         return send_message('Please specify a speaker.', 403)


@app.route('/load_sentence', methods=['POST'])
def load_sentence():
    text = request.form.get('text')
    speaker = request.form.get('speaker')
    if text and speaker:
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


@app.route('/modify')
def modify():
    pass


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
    # try:
    deepstory.combine_wavs()
    return send_message(f'Clip created.')
    # except:
    #     return send_message('Unknown Error.', 403)


@app.route("/wav/<int:sentence_id>")
def stream(sentence_id):
    response = make_response(deepstory.stream(sentence_id), 200)
    response.mimetype = "audio/x-wav"
    return response


if __name__ == '__main__':
    app.run()
