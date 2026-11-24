""" Tests for reading data in fixed size windows, independent of how data
is split into storage chunks.
"""

import os
from collections import defaultdict

import numpy as np
import pytest

from nptdms import TdmsFile
from nptdms.base_segment import DEFAULT_READ_WINDOW_SIZE
from nptdms.test import scenarios
from nptdms.test.util import (
    GeneratedFile,
    channel_metadata,
    segment_objects_metadata)


class CountingFile(object):
    """ Wraps a file object and counts the number of bytes read from it
    """
    def __init__(self, file):
        self._file = file
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


def _channel(tdms_file, group_name, channel_name):
    return tdms_file[group_name][channel_name]


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_windows_match_storage_chunk_reading(test_file, expected_data):
    """ Reading via fixed size windows gives the same data as reading whole chunks
    """
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for (group_name, channel_name), expected in expected_data.items():
                channel = _channel(tdms_file, group_name, channel_name)
                np.testing.assert_array_equal(channel.read_data(), expected)


def test_window_count_independent_of_storage_chunk_size():
    """ Within a segment, the number of windows depends on data length and window
    size, not on the number of storage chunks.

    A single segment is created with repeated storage chunks by writing one
    metadata-bearing segment whose raw data contains multiple chunks worth of
    data, so num_chunks > 1 while there is only one segment.
    """
    values = np.arange(DEFAULT_READ_WINDOW_SIZE * 4, dtype=np.int32)
    expected_windows = len(values) // DEFAULT_READ_WINDOW_SIZE

    for rows_per_chunk in (1, 2, DEFAULT_READ_WINDOW_SIZE, len(values)):
        num_chunks = len(values) // rows_per_chunk
        test_file = GeneratedFile()
        metadata = segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, rows_per_chunk))
        # One segment whose raw data area contains several storage chunks;
        # the reader derives num_chunks from the raw data size.
        test_file.add_segment(
            ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
            metadata, values.tobytes(), binary_data=True)
        with test_file.get_tempfile() as temp_file:
            with TdmsFile.open(temp_file.file) as tdms_file:
                assert len(tdms_file._reader._segments) == 1
                segment = tdms_file._reader._segments[0]
                assert segment.supports_windowed_reading
                assert segment.num_chunks == num_chunks
                assert segment.get_num_windows() == expected_windows
                channel = _channel(tdms_file, "group", "channel1")
                np.testing.assert_array_equal(channel.read_data(), values)


def test_single_index_reads_bounded_bytes():
    """ Reading a single value only reads one window of data
    """
    rows_per_chunk = 100_000
    num_chunks = 4
    num_rows = rows_per_chunk * num_chunks
    data = np.arange(num_rows, dtype=np.int64).tobytes()

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 4, rows_per_chunk)),
        data, binary_data=True)
    for _ in range(num_chunks - 1):
        test_file.add_segment(("kTocRawData",), "", data, binary_data=True)

    with test_file.get_tempfile() as temp_file:
        counting_file = CountingFile(temp_file.file)
        with TdmsFile.open(counting_file) as tdms_file:
            channel = _channel(tdms_file, "group", "channel1")
            for index in (0, num_rows // 2, num_rows - 1):
                counting_file.reset_counter()
                value = channel[index]
                assert value == index
                # Only a single window of 8-byte values should be read, plus a
                # small fixed overhead for checking the segment start tag.
                assert counting_file.bytes_read <= DEFAULT_READ_WINDOW_SIZE * 8 + 16


def test_interleaved_single_index_does_not_read_whole_segment():
    """ Reading one value from interleaved data only reads the containing window
    """
    rows_per_chunk = 100_000
    num_chunks = 4
    num_rows = rows_per_chunk * num_chunks
    num_channels = 3
    interleaved = np.empty(num_rows * num_channels, dtype=np.int32)
    for row in range(num_rows):
        for channel in range(num_channels):
            interleaved[row * num_channels + channel] = row * num_channels + channel

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            *[channel_metadata("/'group'/'channel%d'" % i, 3, rows_per_chunk)
              for i in range(num_channels)]),
        interleaved.tobytes(), binary_data=True)
    for _ in range(num_chunks - 1):
        test_file.add_segment(
            ("kTocRawData", "kTocInterleavedData"), "",
            interleaved.tobytes(), binary_data=True)

    with test_file.get_tempfile() as temp_file:
        counting_file = CountingFile(temp_file.file)
        with TdmsFile.open(counting_file) as tdms_file:
            channel = _channel(tdms_file, "group", "channel1")
            counting_file.reset_counter()
            value = channel[num_rows // 2]
            assert value == (num_rows // 2) * num_channels + 1
            # Interleaved windows read full rows, but bounded by the window size,
            # plus a small fixed overhead for checking the segment start tag.
            max_bytes = DEFAULT_READ_WINDOW_SIZE * num_channels * 4 + 16
            assert counting_file.bytes_read <= max_bytes


def test_slice_across_window_boundary():
    """ A slice spanning multiple windows reads the same values as the whole channel
    """
    rows_per_chunk = DEFAULT_READ_WINDOW_SIZE
    num_rows = rows_per_chunk * 3
    data = np.arange(num_rows, dtype=np.int32).tobytes()

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, rows_per_chunk)),
        data, binary_data=True)
    for _ in range(2):
        test_file.add_segment(("kTocRawData",), "", data, binary_data=True)

    start = DEFAULT_READ_WINDOW_SIZE - 50
    stop = DEFAULT_READ_WINDOW_SIZE * 2 + 50
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            channel = _channel(tdms_file, "group", "channel1")
            expected = np.arange(num_rows, dtype=np.int32)
            np.testing.assert_array_equal(channel[start:stop], expected[start:stop])


def test_streaming_chunk_offsets_are_contiguous():
    """ Chunks streamed from a channel have contiguous offsets that sum to its length
    """
    test_file, expected_data = scenarios.chunked_interleaved_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for (group_name, channel_name), expected in expected_data.items():
                channel = _channel(tdms_file, group_name, channel_name)
                offset = 0
                total = 0
                for chunk in channel.data_chunks():
                    assert chunk.offset == offset
                    offset += len(chunk)
                    total += len(chunk)
                assert total == len(expected)


def test_many_small_storage_chunks_produce_bounded_windows():
    """ A segment with many small storage chunks still produces a bounded number
    of windows rather than one chunk per storage unit.
    """
    rows_per_chunk = 2
    num_chunks = DEFAULT_READ_WINDOW_SIZE * 10
    num_rows = rows_per_chunk * num_chunks
    data = np.arange(num_rows, dtype=np.int32).tobytes()

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, rows_per_chunk),
            channel_metadata("/'group'/'channel2'", 3, rows_per_chunk)),
        np.tile(np.arange(num_rows, dtype=np.int32), 2).tobytes(),
        binary_data=True)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            segment = tdms_file._reader._segments[0]
            assert segment.num_chunks == num_chunks
            assert segment.get_num_windows() < num_chunks
            file_chunk_count = sum(1 for _ in tdms_file.data_chunks())
            assert file_chunk_count == segment.get_num_windows()
            channel = _channel(tdms_file, "group", "channel1")
            channel_chunk_count = sum(1 for _ in channel.data_chunks())
            assert channel_chunk_count == segment.get_num_windows()


def test_interleaved_and_contiguous_layouts_give_identical_data():
    """ The same data written interleaved and contiguous reads back identically
    through the windowed readers.
    """
    rows_per_chunk = 100
    num_chunks = 50
    num_rows = rows_per_chunk * num_chunks
    num_channels = 3
    channel_values = [np.arange(num_rows, dtype=np.int32) + c for c in range(num_channels)]

    contiguous_blocks = []
    interleaved_blocks = []
    for chunk in range(num_chunks):
        start = chunk * rows_per_chunk
        end = start + rows_per_chunk
        for values in channel_values:
            contiguous_blocks.append(values[start:end].tobytes())
        for row in range(start, end):
            for values in channel_values:
                interleaved_blocks.append(values[row:row + 1].tobytes())

    metadata = segment_objects_metadata(
        *[channel_metadata("/'group'/'channel%d'" % i, 3, rows_per_chunk)
          for i in range(num_channels)])

    contiguous_file = GeneratedFile()
    contiguous_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        metadata, b"".join(contiguous_blocks), binary_data=True)

    interleaved_file = GeneratedFile()
    interleaved_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        metadata, b"".join(interleaved_blocks), binary_data=True)

    for test_file in (contiguous_file, interleaved_file):
        with test_file.get_tempfile() as temp_file:
            with TdmsFile.open(temp_file.file) as tdms_file:
                streamed = defaultdict(list)
                for chunk in tdms_file.data_chunks():
                    for group in chunk.groups():
                        for channel_chunk in group.channels():
                            streamed[(group.name, channel_chunk.name)].extend(channel_chunk[:])

        for channel_index, values in enumerate(channel_values):
            np.testing.assert_array_equal(
                streamed[("group", "channel%d" % channel_index)], values)
