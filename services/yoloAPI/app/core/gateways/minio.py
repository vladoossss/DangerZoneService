from minio import Minio


class MinioServer:
    instance = None
    def __init__(
            self,
            server,
            port,
            user,
            psw) -> None:
        
        self._server = server
        self._port = port
        self._user = user
        self._psw = psw
        self.client = self.create_connection()
        MinioServer.instance = self

    def create_connection(self):
        return Minio(
            endpoint=f'{self._server}:{self._port}', 
            access_key=self._user, 
            secret_key=self._psw, 
            secure=False
            )
