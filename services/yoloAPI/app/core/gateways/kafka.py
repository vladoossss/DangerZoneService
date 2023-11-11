from json import dumps
from kafka import KafkaProducer


# from aiokafka import AIOKafkaProducer
# class Kafka:
#     instance = None

#     def __init__(
#         self,
#         topic,
#         port,
#         servers
#     ) -> None:
#         self._topic = topic
#         self._port = port
#         self._servers = servers
#         self.aioproducer = self.create_kafka()
#         Kafka.instance = self

#     def create_kafka(self):
#         loop = asyncio.get_event_loop()
#         return AIOKafkaProducer(
#             loop=loop,
#             bootstrap_servers=f'{self._servers}:{self._port}'
#         )


class Kafka:
    isinstance = None
    def __init__(
        self,
        topic,
        port,
        servers
    ) -> None:
        
        self._topic = topic
        self._port = port
        self._servers = servers
        self.producer = self.create_kafka()
        Kafka.instance = self

    def create_kafka(self):
        return KafkaProducer(
            bootstrap_servers=f'{self._servers}:{self._port}',
            value_serializer=lambda x: dumps(x).encode('utf-8'),
            retries=3, acks='all'
        )
