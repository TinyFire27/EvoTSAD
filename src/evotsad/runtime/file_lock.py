"""Small cross-platform advisory file-lock helper."""

from contextlib import contextmanager
from typing import IO, Iterator


@contextmanager
def locked(stream: IO, *, shared: bool = False) -> Iterator[IO]:
    if stream.seekable():
        stream.seek(0)
    try:
        if __import__("os").name == "nt":
            import msvcrt
            # msvcrt has no shared lock; an exclusive lock is safest here.
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(" ")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield stream
    finally:
        if __import__("os").name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_UN)
