import os
from enum import Enum


class EnvironmentVariables(str, Enum):
    KAFKA_TOPIC_NAME = 'KAFKA_TOPIC_NAME'
    KAFKA_SERVER = 'KAFKA_SERVER'
    KAFKA_PORT = 'KAFKA_PORT'

    MINIO_HOST="MINIO_HOST"
    MINIO_PORT="MINIO_PORT"
    MINIO_USER="MINIO_USER"
    MINIO_PASSWORD="MINIO_PASSWORD"

    def get_env(self, variable=None):
        return os.environ.get(self, variable)
