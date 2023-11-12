import logging
from json import loads, load

from app.enum import EnvironmentVariables as EnvVariables

from kafka import KafkaConsumer
from minio import Minio

import telebot
bot = telebot.TeleBot(EnvVariables.TELETOKEN.get_env())

def main():
    try:
        # To consume latest messages and auto-commit offsets
        consumer = KafkaConsumer(
            EnvVariables.KAFKA_TOPIC_NAME.get_env(),
            bootstrap_servers=f'{EnvVariables.KAFKA_SERVER.get_env()}:{EnvVariables.KAFKA_PORT.get_env()}',
            value_deserializer=lambda x: loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=False,
        )

        minio = Minio(
            endpoint=f'{EnvVariables.MINIO_HOST.get_env()}:{EnvVariables.MINIO_PORT.get_env()}', 
            access_key=EnvVariables.MINIO_USER.get_env(), 
            secret_key=EnvVariables.MINIO_PASSWORD.get_env(), 
            secure=False
            )

        for message in consumer:

            with open("/consumer/app/volume/pseudo_db.json", "r") as jsonFile:
                data = load(jsonFile)

            for i in data['users']:
                # bot.send_message(i, "%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                #                                  message.offset, message.key, message.value))
            
            
                if "image_name" in message.value.keys():
                    
                    bucket = 'events'
                    im_name = message.value['image_name']
                    max_percent = message.value['max_percent']
                    text_message = message.value['text_message']
                    camera_name = message.value['camera_name']

                    bot.send_message(i, f'{text_message} | On camera {camera_name}')

                    try:
                        response = minio.get_object(bucket, im_name)
                        im_bytes = response.read()
                        bot.send_photo(i, photo=im_bytes)   
                        response.close()
                        response.release_conn()
                    except Exception as e:
                        print(f"Error with getting image from minio {e}")
                    
                else:
                    bot.send_message(i, f"Didnt find image_name in json: {message.value}")
                    print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                                 message.offset, message.key, message.value))

    except Exception as e:
        logging.info('Connection successful', e)
