"""Counting file object used to measure actual bytes read from disk.

Wraps a real binary file and counts every byte returned by read() and
readinto(), so the measured number reflects what nptdms actually pulls
from the file rather than its on-disk size.
"""

import os


class CountingFile:
    def __init__(self, path):
        self._file = open(path, 'rb')
        self.bytes_read = 0

    def reset_counter(self):
        self.bytes_read = 0

    def read(self, size=-1):
        data = self._file.read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer):
        count = self._file.readinto(buffer)
        self.bytes_read += count
        return count

    def seek(self, offset, whence=os.SEEK_SET):
        return self._file.seek(offset, whence)

    def tell(self):
        return self._file.tell()

    def write(self, data):
        return self._file.write(data)

    def close(self):
        return self._file.close()

    def fileno(self):
        return self._file.fileno()

    @property
    def name(self):
        return self._file.name
