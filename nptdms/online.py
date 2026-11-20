""" Online reading of TDMS files that are still being written

    This module implements an explicitly opt-in online reading mode for
    TDMS files that an acquisition process is still appending to.
    The regular offline entry points (:meth:`TdmsFile.read` and
    :meth:`TdmsFile.open`) are unchanged and keep their offline semantics.

    Online reading is driven by the main data file only. A segment is
    published once its lead-in, metadata and raw data are all present in
    the main file and the lead-in contains a valid, finite next segment
    offset. A segment that is only partially written, or whose end position
    has not been published yet, is left out of the current view and picked
    up by a later refresh. Any ``.tdms_index`` file is ignored, so an index
    that is written before or after the data file cannot add or remove
    segments from a view.
"""

from collections import OrderedDict
import os
import struct

from nptdms.common import toc_properties
from nptdms.log import log_manager
from nptdms.reader import TdmsReader, ObjectMetadata
from nptdms.tdms_segment import TdmsSegment, SegmentIndexCache


log = log_manager.get_logger(__name__)
_struct_unpack = struct.unpack

_LEAD_IN_LENGTH = 28
_UNKNOWN_NEXT_SEGMENT_OFFSET = 0xFFFFFFFFFFFFFFFF


class OnlineTdmsError(Exception):
    """ Raised when an online refresh cannot deliver a consistent view

    This is raised when the data file was replaced by a different file or
    was truncated, or when a segment that is fully within the published
    region is structurally corrupt. When it is raised, no view is delivered
    and previously delivered views are unaffected.
    """


class OnlineTdmsFile(object):
    """ Reads a TDMS file that is still being written, one published view at a time

    Create an instance with :meth:`nptdms.TdmsFile.open_online`, then call
    :meth:`refresh` to get a :class:`nptdms.TdmsFile` that sees all segments
    that were completely written at the time of the refresh::

        with TdmsFile.open_online(path) as online_file:
            view = online_file.refresh()
            # ... writer appends more segments ...
            with online_file.refresh() as new_view:
                # Use new_view
                ...

    Each refresh returns a new, independent view. Views returned by earlier
    refreshes keep seeing the segments that were published at their own
    refresh, because every view holds its own metadata snapshot and its own
    file handle. Closing this object stops refreshing but does not close
    views that were already delivered; close each view separately when it is
    no longer needed (views also work as context managers).
    """

    def __init__(self, file, raw_timestamps=False, memmap_dir=None):
        """ Initialise a new OnlineTdmsFile

        :param file: Path to the tdms file to read as a string or
            pathlib.Path. The file must exist but may be empty.
        :param raw_timestamps: By default TDMS timestamps are read as numpy
            datetime64 but this loses some precision.
            Setting this to true will read timestamps as a custom
            TdmsTimestamp type.
        :param memmap_dir: The directory to store memory mapped data files
            in, or None to read data into memory.
        """
        if hasattr(file, "read"):
            raise ValueError(
                "Online reading requires a path to a TDMS file, not an already opened file")
        self._reader = OnlineTdmsReader(file)
        self._raw_timestamps = raw_timestamps
        self._memmap_dir = memmap_dir

    def refresh(self):
        """ Advance to the current publish boundary and return a new view of the file

        Reads any newly completed segment metadata and returns a new
        :class:`nptdms.TdmsFile` containing exactly the segments that were
        completely written in the main data file at this point. Channel data
        is still read lazily from disk when it is accessed.

        If no new segments were completed since the previous refresh, the
        returned view sees the same segments as the previous one.

        :rtype: nptdms.TdmsFile
        :raises OnlineTdmsError: If the file was replaced or truncated, or a
            segment within the published region is corrupt. No view is
            delivered in this case and previously delivered views are
            unaffected.
        """
        if self._reader is None:
            raise RuntimeError("Cannot refresh after the online TDMS reader is closed")
        view_reader = self._reader.refresh()
        try:
            from nptdms.tdms import TdmsFile
            return TdmsFile._from_reader(
                view_reader, raw_timestamps=self._raw_timestamps, memmap_dir=self._memmap_dir)
        except Exception:
            view_reader.close()
            raise

    def close(self):
        """ Stop reading the file; further refreshes will raise an error

        No file handle is held between refreshes, so this only prevents
        further refreshes. Views that were already delivered keep their own
        file handles and remain usable until they are closed themselves.
        """
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OnlineTdmsReader(TdmsReader):
    """ Incrementally discovers published segments in a growing TDMS file

    Discovery is driven only by the main data file and obeys the publish
    boundary: a segment is discovered only when its lead-in, metadata and
    raw data are all present and the lead-in declares a finite end position.
    Discovery stops at the first segment that is not publishable yet, so
    segments before it stay readable while the incomplete tail is left for
    the next refresh.
    """

    def __init__(self, file_path):
        """ Initialise a new OnlineTdmsReader

        :param file_path: Path to the tdms file to read.
        """
        self._file_path = str(file_path)
        self._index_file_path = None
        self._file = None
        self._index_file = None

        self._segments = []
        self._prev_segment_objects = {}
        self.object_metadata = OrderedDict()
        self._segment_channel_offsets = {}
        self.tdms_version = None

        self._file_identity = None
        self._file_size = 0
        self._data_file_size = None
        self._next_segment_position = 0
        self._index_cache = SegmentIndexCache()
        self._closed = False
        with self._open_file() as file:
            self._check_file_state(file)

    def refresh(self):
        """ Discover newly published segments and return a frozen view reader

        :rtype: OnlineViewReader
        """
        if self._closed:
            raise RuntimeError("Cannot refresh after the underlying TDMS reader is closed")
        # The file is re-opened for every refresh. A previously opened buffered
        # reader could serve stale bytes for lead-ins that the writer backfilled
        # since the last refresh, and within one refresh discovery only reads
        # forwards, so a fresh file object always sees the current file contents.
        with self._open_file() as file:
            self._check_file_state(file)
            self._discover_segments(file)
        return self._build_view_reader()

    def close(self):
        self._closed = True

    def _open_file(self):
        return open(self._file_path, "rb")

    def _check_file_state(self, file):
        """ Verify that the file is the same file we started reading and only grew
        """
        file_stat = os.fstat(file.fileno())
        file_identity = (file_stat.st_dev, file_stat.st_ino)
        if self._file_identity is None:
            self._file_identity = file_identity
        elif file_identity != self._file_identity:
            raise OnlineTdmsError(
                "The TDMS file at '%s' was replaced by a different file" % self._file_path)
        if file_stat.st_size < self._file_size:
            raise OnlineTdmsError(
                "The TDMS file at '%s' was truncated from %d to %d bytes" % (
                    self._file_path, self._file_size, file_stat.st_size))
        self._file_size = file_stat.st_size

    def _discover_segments(self, file):
        """ Parse segments from the unpublished region, stopping at the publish boundary
        """
        while True:
            position = self._next_segment_position
            if self._file_size - position < _LEAD_IN_LENGTH:
                # The segment lead-in has not been fully written yet
                return
            lead_in = self._read_lead_in(file, position)
            if lead_in is None:
                # The segment end position is unknown or its data has not
                # been fully written yet
                return
            (toc_mask, data_position, next_segment_position) = lead_in
            segment = TdmsSegment(position, toc_mask, next_segment_position, data_position, False)
            previous_segment = self._segments[-1] if self._segments else None
            try:
                properties = segment.read_segment_objects(
                    file, self._prev_segment_objects, self._index_cache, previous_segment)
            except Exception as error:
                raise OnlineTdmsError(
                    "Segment at position %d is corrupt: %s" % (position, error)) from error
            if (toc_mask & toc_properties['kTocMetaData']) and file.tell() != data_position:
                raise OnlineTdmsError(
                    "Segment at position %d is corrupt: segment metadata does not end at "
                    "the raw data offset declared in the segment lead-in" % position)
            self._update_object_properties(properties)
            self._set_segment_scaling_properties(segment)
            self._update_object_metadata(segment)
            self._segments.append(segment)
            self._next_segment_position = next_segment_position

    def _read_lead_in(self, file, position):
        """ Read a segment lead-in

        :return: None if the segment is not publishable yet, otherwise a tuple of
            (toc_mask, data_position, next_segment_position).
        """
        file.seek(position)
        lead_in_bytes = file.read(_LEAD_IN_LENGTH)
        # The caller checked that a full lead-in is present
        tag = lead_in_bytes[:4]
        if tag != b'TDSm':
            raise OnlineTdmsError(
                "Segment at position %d does not start with b'TDSm'" % position)

        toc_mask = _struct_unpack('<l', lead_in_bytes[4:8])[0]
        endianness = '>' if (toc_mask & toc_properties['kTocBigEndian']) else '<'
        (version, next_segment_offset, raw_data_offset) = _struct_unpack(
            endianness + 'lQQ', lead_in_bytes[8:28])

        if self.tdms_version is None:
            if version not in (4712, 4713):
                log.warning("Unrecognised version number: %d" % version)
            self.tdms_version = version
        elif self.tdms_version != version:
            log.warning("Segment version mismatch, %d != %d" % (version, self.tdms_version))

        if next_segment_offset == _UNKNOWN_NEXT_SEGMENT_OFFSET:
            # The writer has not yet published where this segment ends
            return None
        data_position = position + _LEAD_IN_LENGTH + raw_data_offset
        next_segment_position = position + _LEAD_IN_LENGTH + next_segment_offset
        if next_segment_position > self._file_size:
            # The segment metadata or data has not been fully written yet
            return None
        if data_position > next_segment_position:
            raise OnlineTdmsError(
                "Segment at position %d is corrupt: raw data offset %d exceeds the "
                "segment end offset %d" % (position, raw_data_offset, next_segment_offset))
        return toc_mask, data_position, next_segment_position

    def _build_view_reader(self):
        """ Snapshot the currently published segments into a frozen view reader
        """
        # Segments are immutable once published, so views can safely share them.
        # Object metadata is still mutated as new segments are discovered, so
        # each view gets its own copy.
        segments = list(self._segments)
        object_metadata = OrderedDict(
            (path, _copy_object_metadata(metadata))
            for (path, metadata) in self.object_metadata.items())
        view_file = self._open_file()
        try:
            view_stat = os.fstat(view_file.fileno())
            if (view_stat.st_dev, view_stat.st_ino) != self._file_identity:
                raise OnlineTdmsError(
                    "The TDMS file at '%s' was replaced by a different file" % self._file_path)
        except Exception:
            view_file.close()
            raise
        return OnlineViewReader(view_file, segments, object_metadata, self.tdms_version)


class OnlineViewReader(TdmsReader):
    """ A TdmsReader that exposes one frozen set of published segments

    Segment discovery and the channel offset index of this reader are bounded
    by the publish boundary of the refresh that created it. The view owns its
    own file handle, so it keeps working after the online reader that produced
    it has moved on or been closed, until the view itself is closed.
    """

    def __init__(self, file, segments, object_metadata, tdms_version):
        self._file_path = None
        self._index_file_path = None
        self._file = file
        self._index_file = None

        self._segments = segments
        self._prev_segment_objects = {}
        self.object_metadata = object_metadata
        self._segment_channel_offsets = {}
        self.tdms_version = tdms_version
        self._data_file_size = None

    def read_metadata(self, require_segment_indexes=False):
        # The metadata of this view was read when the view was published,
        # so there is nothing to do here.
        pass

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _copy_object_metadata(metadata):
    copied = ObjectMetadata()
    copied.properties = OrderedDict(metadata.properties)
    copied.data_type = metadata.data_type
    copied.scaler_data_types = metadata.scaler_data_types
    copied.num_values = metadata.num_values
    return copied
