from app.core.gateways.minio import MinioServer

def get_minio_instance():
    if MinioServer.instance:
        return MinioServer.instance
    return MinioServer()
