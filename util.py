# SIU KING WAI SM4701 Deepstory
import re
import copy
import spacy

from unidecode import unidecode


def quote_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "“" or token.text == "”":
            doc[token.i + 1].is_sent_start = True
    return doc


nlp = spacy.load('en_core_web_sm')
nlp_no_comma = copy.deepcopy(nlp)
nlp.add_pipe(quote_boundaries, before="parser")
sentencizer = nlp.create_pipe("sentencizer")
sentencizer.punct_chars.add(',')
sentencizer_no_comma = nlp_no_comma.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)
nlp_no_comma.add_pipe(sentencizer_no_comma, first=True)


def normalize_text(text):
    """Normalize text so that some punctuations that indicate pauses will be replaced as commas"""
    replace_list = [
        [r'(\w)’(\w)', r"\1'\2"],  # fix apostrophe for content from books
        [r'(\.\.\.)$|…$', '.'],
        [r'\(|\)|:|;| “|(\s*-+\s+)|(\s+-+\s*)|\s*-{2,}\s*|(\.\.\.)|…|—', ', '],
        [r'\s*,[^\w]*,\s*', ', '],  # capture multiple commas
        [r'\s*,\s*', ', '],  # format commas
        [r'\.,', '.'],
        [r'[‘’“”]', '']  # strip quote
    ]
    for regex, replacement in replace_list:
        text = re.sub(regex, replacement, text)
    text = re.sub(r' +', ' ', text)
    text = unidecode(text)  # Get rid of the accented characters
    return text


def separate(text, n_gram, comma, max_len=30):
    _nlp = nlp if comma else nlp_no_comma
    lines = []
    line = ''
    counter = 0
    for sent in _nlp(text).sents:
        if sent.text:
            if counter == 0:
                line = sent.text
            else:
                line = f'{line} {sent.text}'
            counter += 1

            if counter == n_gram:
                lines.append(_nlp(line))
                line = ''
                counter = 0

    # for remaining sentences
    if line:
        lines.append(_nlp(line))

    return lines
