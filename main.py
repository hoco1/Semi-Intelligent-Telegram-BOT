from telegram.ext import *
from io import BytesIO
import numpy as np
import cv2
from tensorflow import keras
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import random
import json
from tensorflow.keras.models import load_model
import nltk

# load pickle - chatbot
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
intent_dict = pickle.load(open('intent_dict.pkl','rb'))
model_chatbot = load_model('chatbot_model.h5')

# objects
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

with open('token.txt', 'r') as f:
    token = f.read()
    
def start(update,context):
    update.message.reply_text("Hello, I am a bot that can help you to classify the news is fake or not and also \nTell you the name of the objects \n")

def help(update,context):
    update.message.reply_text("""
                              /start - start the bot
/help - show this message
Brief Explanation
First mission : when you send a photo, I'll predict the names of objects
I know these objects' aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
Second mission : when you send a text, I'll predict the news is fake or not
                              """)
    
# load the model and vectorizer - label_encoder.pickle and vectorizer.pickle and svc_news_detection.pickle
vec_file = open('vectorizer.pickle', 'rb')
vec = pickle.load(vec_file)
vec_file.close()

le_file = open('label_encoder.pickle', 'rb')
le = pickle.load(le_file)
le_file.close()

svc_file = open('svc_news_detection.pickle', 'rb')
svc = pickle.load(svc_file)
svc_file.close()

# preprocessing
def text_preprocessing(text):
    text = text.lower()
    # remove punctuation
    text = [letter for letter in text if letter not in string.punctuation]
    # join the list of characters into a string
    text = ''.join(text)
    # remove stopwords
    text = [word for word in text.split() if word not in stopwords.words('english')]
    if len(text) >= 36:
        pass
        # text = [word for word in text.split() if word not in stopwords.words('english')]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


# chat bot function
def bag_of_words(sentence):
    sentence = text_preprocessing(sentence)
    bag = [0]*len(words)
    for w in sentence:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model_chatbot.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        print(r)
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def handle_message(update,context):
    text = update.message.text
    
    if len(text) <= 36:
        ints = predict_class(text)
        update.message.reply_text(intent_dict[ints[0]['intent']][random.randint(0,len(intent_dict[ints[0]['intent']])-1)])
    else:
        text = text_preprocessing(text)
        text = [' '.join(text)]
        text_vec = vec.transform(text)
        predict = str(le.inverse_transform(svc.predict(text_vec)))
        update.message.reply_text("It's probably {}".format(predict))    

def handle_photo(update,context):
    update.message.reply_text("I am processing your image")
    
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32),interpolation=cv2.INTER_AREA)
    
    # load model - object_detection_model.h5
    reconstructed_model = keras.models.load_model('object_detection_model.h5')
    
    prediction = reconstructed_model.predict(np.array([img/255]))
    
    update.message.reply_text("It's probably {}".format(class_name[np.argmax(prediction)]))

# use proxy to connect to the server   
REQUEST_KWARGS = { 'proxy_url': 'socks5h://127.0.0.1:9150' }

updater = Updater(token, use_context=True,request_kwargs=REQUEST_KWARGS)
dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))
dp.add_handler(CommandHandler('help', help))

dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()