from telegram.ext import *
from io import BytesIO
import numpy as np
import cv2
from tensorflow import keras
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

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
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

def handle_message(update,context):
    
    text = update.message.text
    text = text_preprocessing(text)
    text = [' '.join(text)]
    text_vec = vec.transform(text)
    
    if len(text) <= 36:
        update.message.reply_text("ChatBOT")
    else:
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