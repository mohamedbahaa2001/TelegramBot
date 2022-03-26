import cv2
from telegram.ext import *
import numpy as np
from io import BytesIO
import tensorflow as tf

token = "" # put your telegram token here
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


def start(update, context):
    update.message.reply_text("Welcome Bro, This bot is fully made by Mohamed Bahaa the genius :D")


def help(update, context):
    update.message.reply_text("""
    /start - starts conversation
    /help - shows this message
    /train - Trains the nural netwrok
    """)


def train(update, context):
    update.message.reply_text("Model is being trained....")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save('cifar_calssifier.model')
    update.message.reply_text("Done training you can now send a photo :D")


def handle_message(update, context):
    update.message.reply_text("please train the model and send a picture! ")


def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_byte = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f"In this image I see a {class_names[np.argmax(prediction)]}")


updater = Updater(token, use_context=True)
db = updater.dispatcher

db.add_handler(CommandHandler("start", start))
db.add_handler(CommandHandler("help", help))
db.add_handler(CommandHandler("train", train))
db.add_handler(MessageHandler(Filters.text, handle_message))
db.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()
