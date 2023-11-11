import telebot
import json
import os 

bot = telebot.TeleBot(os.environ.get("TELETOKEN"))

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):

    with open("volume/pseudo_db.json", "r") as jsonFile:
        data = json.load(jsonFile)
    
    data["users"].append(message.from_user.id)

    with open("volume/pseudo_db.json", "w") as jsonFile:
        json.dump(data, jsonFile)

    bot.reply_to(message, "start")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.from_user.id)

bot.infinity_polling()