from io import UnsupportedOperation
import numpy as np

from nptdms.log import log_manager


log = log_manager.get_logger(__name__)

# Number of values per channel to read at once when streaming data.
# This is independent of how the TDMS file splits data into storage chunks,
# so that the amount of data read depends on the number of values requested
# rather than the storage chunk layout.
DEFAULT_READ_WINDOW_SIZE = 4096


class BaseSegmentObject(object):
    """ Abstract base class for an object in a TDMS segment
    """

    __slots__ = [
        'path', 'number_values', 'data_size',
        'has_data', 'data_type']

    def __init__(self, path):
        self.path = path
        self.number_values = 0
        self.data_size = 0
        self.has_data = False
        self.data_type = None

    def read_raw_data_index(self, file, raw_data_index_header, endianness):
        """ Read the raw data index for a single object in a segment
        """
        raise NotImplementedError("Segment metadata reading must be implemented in base classes")

    @property
    def scaler_data_types(self):
        return None

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.path)


class BaseDataReader(object):
    """ Abstract base class for reading data in a segment
    """

    def __init__(self, num_chunks, final_chunk_lengths_override, endianness, data_position=0):
        self.num_chunks = num_chunks
        self.final_chunk_lengths_override = final_chunk_lengths_override
        self.endianness = endianness
        self.data_position = data_position

    def read_data_chunks(self, file, data_objects, num_chunks):
        """ Read multiple data chunks at once
            In the base case we read each chunk individually but subclasses can override this
        """
        for chunk in range(num_chunks):
            yield self._read_data_chunk(file, data_objects, chunk)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        """ Read data from a chunk for all channels
        """
        raise NotImplementedError("Data chunk reading must be implemented in base classes")

    def read_channel_data_chunks(self, file, data_objects, channel_path, chunk_offset, stop_chunk):
        """ Read multiple data chunks for a single channel at once
            In the base case we read each chunk individually but subclasses can override this
        """
        for chunk_index in range(chunk_offset, stop_chunk):
            yield self._read_channel_data_chunk(file, data_objects, chunk_index, channel_path)

    def _read_channel_data_chunk(self, file, data_objects, chunk_index, channel_path):
        """ Read data from a chunk for a single channel
        """
        # In the base case we can read data for all channels
        # and then select only the requested channel.
        # Derived classes can implement more optimised reading.
        data_chunk = self._read_data_chunk(file, data_objects, chunk_index)
        return data_chunk_to_channel_chunk(data_chunk, channel_path)

    def read_windows(self, file, data_objects, window_offset, stop_window, window_size):
        """ Read data in fixed size windows, independent of how data is split into storage chunks

        :returns: Generator of RawDataChunk objects, each containing up to window_size
            values for each channel that has data in the window.
        """
        for window_index in range(window_offset, stop_window):
            yield self._read_window(file, data_objects, window_index, window_size)

    def read_channel_windows(self, file, data_objects, channel_path, window_offset, stop_window, window_size):
        """ Read data for a single channel in fixed size windows

        :returns: Generator of RawChannelDataChunk objects, each containing up to
            window_size values for the requested channel.
        """
        for window_index in range(window_offset, stop_window):
            yield self._read_channel_window(file, data_objects, channel_path, window_index, window_size)

    def _read_window(self, file, data_objects, window_index, window_size):
        """ Read a fixed size window of data for all channels
        """
        raise NotImplementedError("Windowed data reading must be implemented in derived classes")

    def _read_channel_window(self, file, data_objects, channel_path, window_index, window_size):
        """ Read a fixed size window of data for a single channel
        """
        data_chunk = self._read_window(file, data_objects, window_index, window_size)
        return data_chunk_to_channel_chunk(data_chunk, channel_path)


class RawDataChunk(object):
    """Data read from a single chunk in a TDMS segment

    :ivar channel_data: A dictionary of channel data chunks.
        Keys are object paths and values are RawChannelDataChunk instances.
    """

    def __init__(self, channel_data):
        self.channel_data = channel_data

    @staticmethod
    def empty():
        return RawDataChunk({})

    @staticmethod
    def channel_data(data):
        channel_chunks = {
            path: RawChannelDataChunk.channel_data(d)
            for (path, d) in data.items()
        }
        return RawDataChunk(channel_chunks)

    @staticmethod
    def scaler_data(data):
        channel_chunks = {
            path: RawChannelDataChunk.scaler_data(d)
            for (path, d) in data.items()
        }
        return RawDataChunk(channel_chunks)


class RawChannelDataChunk(object):
    """Data read for a single channel from a single chunk in a TDMS segment

    :ivar data: Raw data in this chunk for a standard TDMS channel.
    :ivar scaler_data: A dictionary of scaler data in this segment for
        DAQmx raw data. Keys are the scaler id and values are data arrays.
    """

    def __init__(self, data, scaler_data):
        self.data = data
        self.scaler_data = scaler_data

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        elif self.scaler_data is not None:
            return next(len(d) for d in self.scaler_data.values())
        return 0

    @staticmethod
    def empty():
        return RawChannelDataChunk(None, None)

    @staticmethod
    def channel_data(data):
        return RawChannelDataChunk(data, None)

    @staticmethod
    def scaler_data(data):
        return RawChannelDataChunk(None, data)


def fromfile(file, dtype, count, *args, **kwargs):
    """Read from any file-like object into a numpy array"""

    itemsize = np.dtype(dtype).itemsize
    buffer = np.zeros(count * itemsize, np.uint8)
    offset = 0
    while offset < buffer.size:
        bytes_read = file.readinto(buffer[offset:])
        if bytes_read == 0:
            # End of file, don't try to read any more data
            break
        offset += bytes_read
    rounded_bytes = (offset // itemsize) * itemsize
    buffer = buffer[:rounded_bytes]
    buffer.dtype = dtype
    return buffer


def read_interleaved_segment_bytes(f, bytes_per_row, num_values):
    """ Read a segment of interleaved data as rows of bytes
    """
    number_bytes = bytes_per_row * num_values
    combined_data = fromfile(f, dtype=np.uint8, count=number_bytes)

    try:
        # Reshape, so that one row is all bytes for all objects
        combined_data = combined_data.reshape(-1, bytes_per_row)
    except ValueError:
        # Probably incomplete segment at the end => try to clip data
        crop_len = (combined_data.shape[0] // bytes_per_row)
        crop_len *= bytes_per_row
        log.warning("Cropping data from %d to %d bytes to match segment "
                    "size derived from channels",
                    combined_data.shape[0], crop_len)
        combined_data = combined_data[:crop_len].reshape(-1, bytes_per_row)
    return combined_data


def data_chunk_to_channel_chunk(data_chunk, channel_path):
    try:
        return data_chunk.channel_data[channel_path]
    except KeyError:
        return RawChannelDataChunk.empty()
