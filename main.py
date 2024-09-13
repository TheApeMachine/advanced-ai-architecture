import asyncio
import capnp
from logger import Logger
import artifact_capnp

logger = Logger(name='ailogger')

class ModelServiceImpl(artifact_capnp.ModelService.Server):

    def __init__(self, index):
        self.index = index
        logger.info("ModelServiceImpl initialized with Model")

    async def query(self, request, **kwargs):  # Make the method asynchronous
        logger.debug(f"Received query: {request.question}")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(request.question)
        logger.debug(f"Query response: {response}")
        return artifact_capnp.Artifact.new_message(question=str(response))

async def new_connection(stream, index):
    logger.info("New connection established")
    
    # Pass the pre-loaded index to the service implementation
    await capnp.TwoPartyServer(stream, bootstrap=ModelServiceImpl(index)).on_disconnect()
    logger.info("Connection closed")


async def main():
    server = await capnp.AsyncIoStream.create_server(
        lambda stream: new_connection(stream), "0.0.0.0", "8123"
    )
    
    logger.info(f"Server started and listening on 0.0.0.0:8123")
    
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(capnp.run(main()))
    except Exception as e:
        logger.error(f"An error occurred: {e}")