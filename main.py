from telegram.ext import *
from io import BytesIO
import numpy as np
import cv2

with open('token.txt', 'r') as f:
    token = f.read()
    
def start(update,context):
    update.message.reply_text("Hello, I am a bot that can help you to classify the news is fake or not and also \nTell you the name of the objects \n")

def help(update,context):
    update.message.reply_text("You can type /start to see the help")

def train(update,context):
    pass

def handle_message(update,context):
    update.message.reply_text("ok")

def handle_photo(update,context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32),interpolation=cv2.INTER_AREA)
    
    update.message.reply_text("I am processing your image")

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