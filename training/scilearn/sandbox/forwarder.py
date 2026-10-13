"""Relay TCP connections on 127.0.0.1:<port> to the host egress proxy's Unix socket.

Runs inside the sandbox (no network namespace access beyond loopback). Policy is
enforced by the host proxy, not here.
"""
import asyncio
import sys


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(cr, cw, sock_path):
    try:
        ur, uw = await asyncio.open_unix_connection(sock_path)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main(host, port, sock_path):
    server = await asyncio.start_server(lambda r, w: handle(r, w, sock_path), host, port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
