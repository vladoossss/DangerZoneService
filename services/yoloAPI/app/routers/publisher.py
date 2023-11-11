from app.core.gateways.kafka import Kafka
from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance

from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/producer",
    tags=["kafka"],
    dependencies=[Depends(get_kafka_instance)]
    )


@router.post("")
async def check_kafka_connection(data: Message, server: Kafka = Depends(get_kafka_instance)):
    try:
        topic_name = server._topic
        server.producer.send(topic_name, data.dict())
    except Exception as e:
        raise e
    return 'Message sent successfully'
