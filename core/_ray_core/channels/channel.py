import asyncio
import ray
from ray import ObjectRef
from ray.experimental.channel import Channel

class ChannelBroker:
    def __init__(
            self,
            self_ref,
            host,
            neighbor_refs:list[str, ObjectRef]
    ):
        self.host=host
        self.neighbor_refs=neighbor_refs
        # Create a bounded channel (prevents OOM)
        # Channel
        self.reader_nlist = [
            (nid, ref)
            for nid, ref in neighbor_refs
        ]

        self.channel = Channel(
            writer=self_ref,
            reader_and_node_list=self.reader_nlist,
        )

    def get_writer(self):
        """Return the writer endpoint for producers."""
        return self.channel.writer

    def get_reader(self):
        """Return the reader endpoint for the collector."""
        return self.channel.reader


@ray.remote
class DataProducer:
    def __init__(self, broker_writer):
        self.writer = broker_writer

    async def send(self, payload):
        # Use ray.put for large data
        await self.writer.write(payload)


@ray.remote
class DataCollector:
    def __init__(self, broker_reader):
        self.reader = broker_reader

    async def run(self):
        """Continuously consume and clear messages in time order."""
        async for msg in self.reader:
            timestamp, data = msg["ts"], msg["data"]
            # Process message
            print(f"[Collector] {timestamp}: {data}")
            # After processing, the message is auto-removed from buffer


# ------------------------
# Example usage
# ------------------------
async def main():


    # 1. Create broker
    broker = await ChannelBroker.remote().__ray_metadata__["actor_handle"]
    writer = await broker.get_writer.remote()
    reader = await broker.get_reader.remote()

    # 2. Start collector
    collector = DataCollector.remote(reader)
    asyncio.create_task(collector.run.remote())

    # 3. Spawn producers
    producers = [DataProducer.remote(writer) for _ in range(5)]

    # 4. Send messages
    import time
    for i in range(10):
        payload = {"ts": time.time(), "data": f"msg-{i}"}
        [await p.send.remote(payload) for p in producers]
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
