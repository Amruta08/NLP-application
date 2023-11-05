from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from textstat import flesch_reading_ease
from googletrans import Translator
import language_tool_python
import speech_recognition as sr
import pyttsx3
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from collections import Counter
stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')


# create instance of flask class
app = Flask(__name__)

tool = language_tool_python.LanguageTool('en-US')
translator = Translator()

# Initialize the recognizer
r = sr.Recognizer()

punctuation = punctuation + '\n'


def sentiment(c):
    if c < 0:
        return 'Negative'
    elif c == 0:
        return 'Neutral'
    else:
        return 'Positive'


def complexity(x):
    if x < 10:
        return 'Professional'
    elif x > 10 and x <= 30:
        return 'College graduate(Very Difficult)'
    elif x > 30 and x <= 50:
        return 'College (Difficult)'
    elif x > 50 and x <= 60:
        return '10th to 12th grade(Fairly difficult)'
    elif x > 60 and x <= 70:
        return '8th & 9th grade(Plain English)'
    elif x > 70 and x <= 80:
        return '7th grade(Fairly easy)'
    elif x > 80 and x <= 90:
        return '6th grade(Easy to read)'
    elif x > 90:
        return '5th grade(Very easy)'


def summary(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Word tokenization
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    # Sentence tokenization
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    # Word frequency
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    print(sentence_scores)

    # Summarization
    select_length = int(len(sentence_tokens)*0.3)

    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary


def keywords(text):
    words = text.split()
    word_counts = Counter(words)
    total_words= len(words)
    
    word_percentage = {word: round((count / total_words), 2) for word, count in word_counts.items()}
    
    for key in word_percentage.copy():
        if key in stopwords:
            word_percentage.pop(key)
            
    new_dict =  dict(Counter(word_percentage).most_common(10))    
    
    return new_dict
    

@app.route("/", methods=['GET', 'POST'])
def index():
    correct = ""
    data = ""
    senti = ""
    fre = ""
    word_count = ""
    char_count = ""
    error_messages = []
    if request.method == 'POST':
        data = request.form['plaintext']
        matches = tool.check(data)
        correct = tool.correct(data)
        s = TextBlob(correct).sentiment.polarity
        senti = sentiment(s)
        a = flesch_reading_ease(correct)
        fre = complexity(a)
        word_count = str(len(correct.split()))
        char_count = len(correct.replace(" ", ""))
        for obj in matches:
            start = int(obj.offset)
            end = int(obj.offset + obj.errorLength)
            a = f'{obj.message} in "{data[start:end]}"'
            error_messages.append(a)

    return render_template("index.html", outp=correct, plain=data, pol=senti, read=fre, wc=word_count, cc=char_count, em=error_messages)


@app.route("/translate", methods=['GET', 'POST'])
def translate():
    translated_text = ""
    data = ""
    senti = ""
    fre = ""
    word_count = ""
    char_count = ""
    if request.method == 'POST':
        data = request.form['plaintext']
        translated_obj = translator.translate(text=data, dest='en')
        translated_text = translated_obj.text
        s = TextBlob(translated_text).sentiment.polarity
        senti = sentiment(s)
        a = flesch_reading_ease(translated_text)
        fre = complexity(a)
        word_count = str(len(translated_text.split()))
        char_count = len(translated_text.replace(" ", ""))
    return render_template("translate.html", outp=translated_text, plain=data, pol=senti, read=fre, wc=word_count, cc=char_count)


@app.route("/summarize", methods=['GET', 'POST'])
def summarize():
    summarize_text = ""
    data = ""
    senti = ""
    fre = ""
    word_count = ""
    char_count = ""
    keyword_obj = ""
    if request.method == 'POST':
        data = request.form['plaintext']
        summarize_text = summary(data)
        s = TextBlob(summarize_text).sentiment.polarity
        senti = sentiment(s)
        a = flesch_reading_ease(summarize_text)
        fre = complexity(a)
        word_count = str(len(summarize_text.split()))
        char_count = len(summarize_text.replace(" ", ""))
        keyword_obj = keywords(data)
    return render_template("summarize.html", outp=summarize_text, plain=data, pol=senti, read=fre, wc=word_count, cc=char_count, obj=keyword_obj)


@app.route('/record_audio', methods=['POST'])
def record_audio():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service; {0}".format(
            e)

    return jsonify({'text': text})

@app.route('/record_audio_hi', methods=['POST'])
def record_audio_hi():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="hi-IN")
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service; {0}".format(
            e)

    return jsonify({'text': text})

@app.route('/record_audio_mr', methods=['POST'])
def record_audio_mr():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="mr-IN")
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service; {0}".format(
            e)

    return jsonify({'text': text})


@app.route('/speak', methods=['POST'])
def speak_text():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    text = request.form.get('text_to_speak')
    engine.say(text)
    engine.runAndWait()
    return "Text has been spoken"


if __name__ == "__main__":
    app.run(debug=True)
