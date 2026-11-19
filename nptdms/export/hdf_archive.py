"""Lossless, reversible archiving of TDMS files into HDF5 files

This module provides a pair of entry points that convert between local TDMS
and HDF5 files while preserving everything needed to rebuild an equivalent
TDMS file without the original source:

* :func:`export_tdms_to_hdf5` archives a TDMS file into a new HDF5 file.
* :func:`restore_tdms_from_hdf5` rebuilds a new TDMS file from an archive.

Unlike :func:`nptdms.export.hdf_export.from_tdms_file` (the implementation
behind :meth:`nptdms.TdmsFile.as_hdf`), the archive stores *raw, unscaled*
sample values, full TDMS type information and timestamps with their complete
64 bit second fractions.  The original TDMS byte stream is not embedded.

Archive format version 1
------------------------

The archive is a plain, self-describing HDF5 file.  Every HDF5 object can be
opened and sliced on its own with h5py.

::

    /
    |-- .tdms_archive_meta/          Hidden group holding archive metadata
    |     |-- format                  Scalar attribute, "nptdms-tdms-archive"
    |     |-- format_version          Scalar attribute, integer format version
    |     |-- tdms_version            TDMS format version (4712 or 4713)
    |     |-- manifest                JSON scalar dataset describing every
    |     |                           TDMS object and its exact data type
    |     `-- .tdms_nul_properties/   Variable length byte datasets holding
    |                                 string properties that contain NUL bytes,
    |                                 keyed by object and property name
    |-- <group link>/...             One HDF5 group per TDMS group
          `-- <channel link>/         One HDF5 group per TDMS channel
                `-- data              The channel's raw samples

TDMS group and channel names are mapped to HDF5 link names with a reversible
escaping rule (see :func:`_escape_name`), so names containing ``/``, quotes or
spaces are never split or mangled.  The original TDMS names are also stored
verbatim in the manifest and as the ``.tdms_archive_name`` attribute.  Root,
group and channel TDMS properties are written verbatim as HDF5 attributes on
the matching object (the archive root, the group and the channel respectively),
apart from string properties containing NUL bytes, which live in the hidden
meta group because HDF5 string attributes cannot represent them.

Channel data layout
~~~~~~~~~~~~~~~~~~~

* Numeric and boolean channels: a chunked HDF5 dataset with the exact raw
  NumPy dtype of the TDMS channel (booleans use 1 byte per value, as in TDMS).
  No scaling is applied, so integer samples, dtype and order are preserved.
* Timestamp channels: a chunked compound dataset with two little endian
  fields, ``second_fractions`` (uint64) and ``seconds`` (int64), stored in the
  TDMS wire order.  This keeps the full 2**-64 second resolution; two
  timestamps differing by a single fraction bit remain distinct.
* String channels: a chunked dataset of variable length uint8 arrays.  Each
  string is stored as its raw UTF-8 bytes, so empty strings, non-ASCII
  characters and embedded NUL bytes round-trip exactly.

Because each dataset is chunked and holds one fixed, contiguous array per
channel, any h5py user can read or slice individual channels directly (for
example ``f[path]['data'][100:200]``) without touching the rest of the
archive.  Only raw values are stored - scaling information lives entirely in
the channel properties, so scaled and unscaled reads can both be reproduced.

Streaming behaviour
~~~~~~~~~~~~~~~~~~~

Data is streamed rather than accumulated.  When exporting an opened TDMS
file, each source decode chunk is handed straight to the HDF5 export layer, so
at most one source chunk of samples is buffered per channel.  When restoring,
the caller can cap the number of samples processed per batch with
``batch_size``; each batch is passed to the existing TDMS writer path and then
released.  Neither side ever holds a buffer the size of a whole long channel.
"""

import json
import os
import tempfile

import numpy as np

from nptdms import types
from nptdms.timestamp import TimestampArray, TdmsTimestamp
from nptdms.writer import ChannelObject, GroupObject, RootObject, TdmsWriter


# Current archive format version.  Bump this when the on-disk layout changes
# in a way older readers cannot understand.
ARCHIVE_FORMAT_VERSION = 1

# Marker values stored alongside the data.
ARCHIVE_FORMAT_NAME = "nptdms-tdms-archive"
META_GROUP = ".tdms_archive_meta"
DATASET_NAME = "data"
NAME_ATTRIBUTE = ".tdms_archive_name"

# TDMS data types that can be archived, mapped to their TDMS type enum values.
# Anything outside this mapping (DAQmx raw data, complex numbers, extended
# floats, ...) raises an UnsupportedArchiveTypeError naming the offending
# object.
_SUPPORTED_TYPE_ENUMS = {
    types.Int8.enum_value,
    types.Int16.enum_value,
    types.Int32.enum_value,
    types.Int64.enum_value,
    types.Uint8.enum_value,
    types.Uint16.enum_value,
    types.Uint32.enum_value,
    types.Uint64.enum_value,
    types.SingleFloat.enum_value,
    types.DoubleFloat.enum_value,
    types.SingleFloatWithUnit.enum_value,
    types.DoubleFloatWithUnit.enum_value,
    types.Boolean.enum_value,
    types.String.enum_value,
    types.TimeStamp.enum_value,
}

# Compound dtype used for timestamp data, in TDMS wire order: fractions first.
_TIMESTAMP_DTYPE = np.dtype([
    ('second_fractions', '<u8'),
    ('seconds', '<i8'),
])

# Number of samples processed per batch when the caller does not specify one.
DEFAULT_BATCH_SIZE = 1000000

# Name of the child group used for string properties containing NUL bytes.
# HDF5 string attributes cannot embed NUL bytes, so these are kept as
# variable length uint8 datasets under the hidden meta group, keyed by the
# owning object's unique escaped path.  This cannot collide with TDMS objects,
# which all live outside the hidden meta group.
_NUL_PROPERTY_GROUP = ".tdms_nul_properties"
_ROOT_PROPERTY_KEY = "%00root%00"


class ArchiveError(Exception):
    """Base class for errors creating or reading TDMS HDF5 archives."""


class UnsupportedArchiveTypeError(ArchiveError):
    """Raised when a TDMS object uses a data type that cannot be archived."""


class ArchiveFormatError(ArchiveError):
    """Raised when an HDF5 file is not a valid, supported TDMS archive."""


def export_tdms_to_hdf5(tdms_file, hdf5_path, chunk_samples=100000):
    """Export a TDMS file to a lossless, reversible HDF5 archive.

    The target path must not already exist; the archive is written to a
    temporary file in the same directory and moved into place only once it is
    complete, so a failure never leaves a partial file at the destination.

    :param tdms_file: A :class:`~nptdms.TdmsFile`.  Both fully read files and
        files opened with :meth:`~nptdms.TdmsFile.open` are supported; the
        latter streams raw data from disk one source chunk at a time.
    :param hdf5_path: Path of the new HDF5 archive to create.
    :param chunk_samples: HDF5 chunk size in samples for each channel dataset.
        It only controls HDF5 storage granularity, never how much source data
        is buffered.
    """
    import h5py

    destination = os.fspath(hdf5_path)
    _require_new_file(destination)

    if tdms_file._reader is None:
        raise ArchiveError(
            "Cannot archive a TDMS file whose underlying reader has been "
            "closed. Re-open the file with TdmsFile.open or TdmsFile.read.")
    if tdms_file._reader.is_index_file_only() or \
            (tdms_file._reader._file is None and
             tdms_file._reader._index_file_path is not None):
        raise ArchiveError(
            "Cannot archive an index file only (.tdms_index): it contains no "
            "raw sample data. Open the matching .tdms data file instead.")

    if tdms_file.file_status.incomplete_final_segment:
        raise ArchiveError(
            "Refusing to archive an incomplete TDMS file: the final segment "
            "was truncated, so its data cannot be archived faithfully")
    _verify_file_is_complete(tdms_file)

    # Validate every object up front so that no unsupported data is written
    # before the problem is reported.
    channel_records = _collect_channel_records(tdms_file)

    target_directory = os.path.dirname(os.path.abspath(destination))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                suffix='.h5.tmp', dir=target_directory, delete=False) as temp_file:
            temp_path = temp_file.name

        with h5py.File(temp_path, 'w', track_order=True) as h5file:
            meta_group = h5file.create_group(META_GROUP)
            meta_group.attrs.create('format', ARCHIVE_FORMAT_NAME)
            meta_group.attrs.create('format_version', ARCHIVE_FORMAT_VERSION)
            meta_group.attrs.create('tdms_version', int(tdms_file.tdms_version))
            nul_group = meta_group.create_group(
                _NUL_PROPERTY_GROUP, track_order=True)

            _write_properties(h5file, tdms_file.properties,
                              nul_group, _ROOT_PROPERTY_KEY)

            manifest_groups = []
            manifest_channels = []

            for group in tdms_file.groups():
                group_group = h5file.create_group(
                    _escape_name(group.name), track_order=True)
                group_group.attrs.create(NAME_ATTRIBUTE, group.name)
                _write_properties(
                    group_group, group.properties,
                    nul_group, _object_key(group.name))
                manifest_groups.append(group.name)

            for record in channel_records:
                channel = record['channel']
                channel_group = _create_channel_group(
                    h5file, channel, record['type_enum'],
                    len(channel), chunk_samples)
                _write_properties(
                    channel_group, channel.properties,
                    nul_group, _object_key(channel.group_name, channel.name))
                manifest_channels.append({
                    'group': channel.group_name,
                    'name': channel.name,
                    'type_enum': record['type_enum'],
                    'length': len(channel),
                })

            manifest = {
                'format': ARCHIVE_FORMAT_NAME,
                'format_version': ARCHIVE_FORMAT_VERSION,
                'tdms_version': int(tdms_file.tdms_version),
                'groups': manifest_groups,
                'channels': manifest_channels,
            }
            meta_group.create_dataset(
                'manifest', shape=(), dtype=h5py.string_dtype(),
                data=json.dumps(manifest))

            _stream_channel_data(tdms_file, h5file, channel_records)

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def restore_tdms_from_hdf5(hdf5_path, tdms_path, batch_size=DEFAULT_BATCH_SIZE):
    """Restore a TDMS file from an HDF5 archive created by this module.

    The target path must not already exist.  The archive is fully validated
    before the output TDMS file is created, and the TDMS file is written to a
    temporary file in the same directory and moved into place on success, so
    failures never leave a partial file at the destination.

    :param hdf5_path: Path to an HDF5 archive produced by
        :func:`export_tdms_to_hdf5`.
    :param tdms_path: Path of the new TDMS file to create.
    :param batch_size: Maximum number of channel samples read from the archive
        and handed to the TDMS writer in a single batch.  Each batch is
        released after being written, so memory use stays bounded regardless
        of channel length.
    """
    import h5py

    destination = os.fspath(tdms_path)
    _require_new_file(destination)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    with h5py.File(os.fspath(hdf5_path), 'r') as h5file:
        manifest = _read_and_validate_manifest(h5file)
        nul_group = h5file[META_GROUP].get(_NUL_PROPERTY_GROUP)
        root_properties = _read_properties(
            h5file, nul_group, _ROOT_PROPERTY_KEY)
        # Validate that every declared group exists before anything is
        # written, so corrupt archives fail without touching the target.
        for group_name in manifest['groups']:
            if _escape_name(group_name) not in h5file:
                raise ArchiveFormatError(
                    "Group data missing from archive for group %r"
                    % group_name)
        # Read all metadata and validate every dataset before opening any
        # output file, so corrupt archives fail without touching the target.
        restore_records = _build_restore_records(h5file, manifest)
        group_properties = {
            group_name: _read_properties(
                h5file[_escape_name(group_name)],
                nul_group, _object_key(group_name))
            for group_name in manifest['groups']
        }
        tdms_version = int(manifest['tdms_version'])

        target_directory = os.path.dirname(os.path.abspath(destination))
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    suffix='.tdms.tmp', dir=target_directory,
                    delete=False) as temp_file:
                temp_path = temp_file.name

            with TdmsWriter(temp_path, version=tdms_version) as writer:
                writer.write_segment([RootObject(root_properties)])

                for group_name in manifest['groups']:
                    writer.write_segment(
                        [GroupObject(group_name, group_properties[group_name])])

                for record in restore_records:
                    _write_channel_batches(
                        writer, record, batch_size, nul_group)

            os.replace(temp_path, destination)
            temp_path = None
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)


def _verify_file_is_complete(tdms_file):
    """Check the underlying file matches its own segment metadata.

    A truncated file can still parse: depending on where the cut falls, the
    reader may keep stale metadata, mis-detect chunk counts or report more
    samples than raw bytes exist.  Cross-check each segment's declared raw
    data region against the space its objects require, and the actual file
    size against the final segment's end position.
    """
    reader = tdms_file._reader
    segments = getattr(reader, '_segments', None)
    file_object = getattr(reader, '_file', None)
    if file_object is None:
        # Index-only or already closed files cannot be verified here.
        return
    if not segments:
        # A non-empty file that yielded no segments was cut through its
        # lead-in or metadata.
        current_position = file_object.tell()
        file_object.seek(0, os.SEEK_END)
        actual_size = file_object.tell()
        file_object.seek(current_position, os.SEEK_SET)
        if actual_size > 0:
            raise ArchiveError(
                "TDMS file is truncated: no complete segment could be read "
                "from %d bytes" % actual_size)
        return
    for segment in segments:
        _verify_segment_data_size(segment)

    current_position = file_object.tell()
    file_object.seek(0, os.SEEK_END)
    actual_size = file_object.tell()
    file_object.seek(current_position, os.SEEK_SET)
    expected_end = segments[-1].next_segment_pos
    if actual_size < expected_end:
        raise ArchiveError(
            "TDMS file is truncated: expected at least %d bytes but the file "
            "contains %d bytes" % (expected_end, actual_size))
    if actual_size > expected_end:
        # Bytes remain after the final complete segment.  A new segment must
        # start with the TDSm tag and have a full 28 byte lead-in, so any
        # smaller remainder is the start of a segment cut off mid-write.
        trailing = actual_size - expected_end
        if trailing < 28:
            raise ArchiveError(
                "TDMS file is truncated: %d trailing byte(s) after the final "
                "complete segment do not form a valid segment" % trailing)
        file_object.seek(expected_end, os.SEEK_SET)
        if file_object.read(4) != b'TDSm':
            raise ArchiveError(
                "TDMS file is truncated: trailing bytes after the final "
                "complete segment do not start with a segment header")
        file_object.seek(current_position, os.SEEK_SET)


def _verify_segment_data_size(segment):
    """Ensure a segment's raw data region can hold its declared samples."""
    if not getattr(segment, 'ordered_objects', None):
        return
    data_objects = [obj for obj in segment.ordered_objects if obj.has_data]
    if not data_objects:
        return
    expected_per_chunk = 0
    for obj in data_objects:
        data_type = obj.data_type
        if data_type is None or getattr(data_type, 'size', None) is None:
            # Variable width (strings) and DAQmx data are not archived and
            # rejected elsewhere; nothing more to check here.
            return
        expected_per_chunk += data_type.size * obj.number_values
    num_chunks = segment.num_chunks
    available = segment.next_segment_pos - segment.data_position
    expected = expected_per_chunk * num_chunks
    if available < expected:
        raise ArchiveError(
            "TDMS segment at byte %d is truncated: its raw data region has %d "
            "bytes but its metadata requires %d"
            % (segment.position, available, expected))


def _collect_channel_records(tdms_file):
    """Build metadata records for every channel, rejecting unsupported types."""
    records = []
    for group in tdms_file.groups():
        for channel in group.channels():
            data_type = channel.data_type
            if data_type is None:
                # A channel that never had data or a data type in the source.
                type_enum = types.Void.enum_value
            else:
                type_enum = data_type.enum_value
            if type_enum not in _SUPPORTED_TYPE_ENUMS:
                raise UnsupportedArchiveTypeError(
                    "Cannot archive channel %s: TDMS data type %s (enum %s) "
                    "is not supported by the reversible HDF5 archive"
                    % (channel.path,
                       data_type.__name__ if data_type is not None else 'Void',
                       type_enum))
            records.append({
                'channel': channel,
                'path': channel.path,
                'group': channel.group_name,
                'name': channel.name,
                'type_enum': type_enum,
            })
    return records


def _create_channel_group(h5file, channel, type_enum, length, chunk_samples):
    """Create the HDF5 group and chunked raw data dataset for a channel."""
    import h5py

    parent = h5file[_escape_name(channel.group_name)]
    channel_group = parent.create_group(
        _escape_name(channel.name), track_order=True)
    channel_group.attrs.create(NAME_ATTRIBUTE, channel.name)

    if length == 0:
        # Empty channels carry no data dataset; their existence and type are
        # recorded in the manifest, and restore writes them as metadata-only
        # objects with their declared type.
        return channel_group

    if type_enum == types.Void.enum_value:
        # Typeless channels have no data to store at all.
        return channel_group

    if type_enum == types.String.enum_value:
        dtype = h5py.vlen_dtype(np.dtype('uint8'))
        chunks = None
    elif type_enum == types.TimeStamp.enum_value:
        dtype = _TIMESTAMP_DTYPE
        chunks = (min(chunk_samples, length),)
    else:
        dtype = _tdms_enum_to_numpy_dtype(type_enum)
        chunks = (min(chunk_samples, length),)

    channel_group.create_dataset(
        DATASET_NAME, shape=(length,), dtype=dtype, chunks=chunks)
    return channel_group


def _stream_channel_data(tdms_file, h5file, channel_records):
    """Write all channel samples, streaming source chunks or in-memory blocks."""
    if tdms_file.data_read:
        _stream_from_loaded_file(h5file, channel_records)
    else:
        _stream_from_open_file(tdms_file, h5file, channel_records)


def _stream_from_loaded_file(h5file, channel_records):
    record_by_path = dict((record['path'], record) for record in channel_records)
    for record in record_by_path.values():
        channel = record['channel']
        if len(channel) == 0 or record['type_enum'] == types.Void.enum_value:
            continue
        dataset = _dataset_for(h5file, record)
        # read_data returns the raw, unscaled array.  Process it in blocks so
        # the staging copy never approaches the full channel length.
        block_size = DEFAULT_BATCH_SIZE
        offset = 0
        total = len(channel)
        while offset < total:
            block = channel.read_data(offset, block_size, scaled=False)
            _write_block(dataset, record['type_enum'], offset, block)
            offset += len(block)
            if len(block) < block_size:
                break


def _stream_from_open_file(tdms_file, h5file, channel_records):
    # Use the existing chunked TDMS reader directly so the samples are raw and
    # unscaled (the public ChannelDataChunk slicing API applies scaling).
    # Each decoded source chunk is written straight to HDF5 and then released,
    # so at most one source decode chunk of samples is held at a time.  The
    # reader's segment index, used by the normal slicing API, is left intact.
    record_by_path = dict((record['path'], record) for record in channel_records)
    offsets = dict((record['path'], 0) for record in channel_records)
    for raw_chunk in tdms_file._reader.read_raw_data():
        for path, channel_chunk in raw_chunk.channel_data.items():
            record = record_by_path.get(path)
            if record is None or channel_chunk.data is None:
                continue
            offset = offsets[path]
            offsets[path] = offset + len(channel_chunk)
            if len(channel_chunk) == 0 or \
                    record['type_enum'] == types.Void.enum_value:
                continue
            dataset = _dataset_for(h5file, record)
            _write_block(
                dataset, record['type_enum'], offset, channel_chunk.data)

    # A truncated or otherwise corrupt source can make the reader stop early,
    # so verify that every channel received exactly its declared length.
    for record in channel_records:
        if record['type_enum'] == types.Void.enum_value or \
                len(record['channel']) == 0:
            continue
        if offsets[record['path']] != len(record['channel']):
            raise ArchiveError(
                "Channel %s is truncated: expected %d samples but read %d"
                % (record['path'], len(record['channel']),
                   offsets[record['path']]))


def _write_block(dataset, type_enum, offset, block):
    """Write one block of raw samples into a channel dataset."""
    end = offset + len(block)
    if type_enum == types.String.enum_value:
        _write_string_block(dataset, offset, block)
    elif type_enum == types.TimeStamp.enum_value:
        dataset[offset:end] = _as_wire_order_timestamps(block)
    else:
        dataset[offset:end] = block


def _write_string_block(dataset, offset, strings):
    """Store strings as individual variable length uint8 byte arrays."""
    for index, value in enumerate(strings):
        encoded = value.encode('utf-8')
        dataset[offset + index] = np.frombuffer(encoded, dtype='uint8')


def _as_wire_order_timestamps(block):
    """Return timestamp raw data using the TDMS wire-order compound dtype."""
    # A TimestampArray may use either field order, and slices can arrive as a
    # plain structured ndarray; either way build an explicit wire-ordered copy
    # of this block only.
    ordered = np.empty(len(block), dtype=_TIMESTAMP_DTYPE)
    ordered['seconds'] = block['seconds']
    ordered['second_fractions'] = block['second_fractions']
    return ordered


def _dataset_for(h5file, record):
    return h5file[_escape_name(record['group'])][
        _escape_name(record['name'])][DATASET_NAME]


def _object_key(group_name, channel_name=None):
    """Stable, collision-free key for an object's NUL string properties."""
    if group_name is None:
        return _ROOT_PROPERTY_KEY
    if channel_name is None:
        return 'g:' + _escape_name(group_name)
    return 'c:' + _escape_name(group_name) + '|' + _escape_name(channel_name)


def _write_properties(h5_object, properties, nul_group, object_key):
    """Write TDMS properties as HDF5 attributes, preserving exact values."""
    object_nul_group = None
    for name, value in properties.items():
        if isinstance(value, str) and '\x00' in value:
            # HDF5 string attributes cannot contain NUL bytes, so keep these
            # as variable length uint8 datasets under the hidden meta group.
            if object_nul_group is None:
                object_nul_group = nul_group.require_group(object_key)
            dataset = object_nul_group.create_dataset(
                _escape_name(name), shape=(),
                dtype=h5py_vlen_uint8())
            dataset[()] = np.frombuffer(value.encode('utf-8'), dtype='uint8')
        else:
            h5_object.attrs.create(name, _encode_property_value(name, value))


def h5py_vlen_uint8():
    import h5py
    return h5py.vlen_dtype(np.dtype('uint8'))


def _read_properties(h5_object, nul_group, object_key):
    """Read TDMS properties back from attributes and NUL string datasets."""
    properties = {}
    for name in h5_object.attrs.keys():
        if name == NAME_ATTRIBUTE:
            continue
        properties[name] = _decode_property_value(h5_object.attrs[name])
    object_nul_group = nul_group.get(object_key) if nul_group is not None else None
    if object_nul_group is not None:
        for link_name in object_nul_group.keys():
            name = _unescape_name(link_name)
            properties[name] = bytes(
                object_nul_group[link_name][()]).decode('utf-8')
    return properties


def _encode_property_value(name, value):
    """Convert a non-NUL TDMS property value into an HDF5 attribute value."""
    if isinstance(value, TdmsTimestamp):
        return np.void(np.array(
            (value.second_fractions, value.seconds), dtype=_TIMESTAMP_DTYPE))
    if isinstance(value, types.TimeStamp):
        # Property values read from TDMS use the lower level timestamp type,
        # which stores the wire bytes rather than split fields.
        parsed = np.frombuffer(
            value.bytes, dtype=_TIMESTAMP_DTYPE)[0]
        return np.void(parsed)
    if isinstance(value, np.datetime64):
        # Timestamps read without raw_timestamps arrive as datetime64; convert
        # back through the same encoding the TDMS writer uses.
        return _encode_property_value(name, types.TimeStamp(value))
    if isinstance(value, np.ndarray):
        # TDMS properties are scalars; keep numpy scalar types intact.
        if value.shape == ():
            return value
        raise UnsupportedArchiveTypeError(
            "Cannot archive array-valued property %r" % name)
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    if isinstance(value, int):
        # Preserve width when a typed scalar was used.
        return value
    if isinstance(value, float):
        return np.float64(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return np.bytes_(value)
    raise UnsupportedArchiveTypeError(
        "Cannot archive property %r with value type %s"
        % (name, type(value).__name__))


def _decode_property_value(value):
    if isinstance(value, np.void):
        # Compound scalar: a TDMS timestamp.
        return TdmsTimestamp(
            int(value['seconds']), int(value['second_fractions']))
    if isinstance(value, bytes):
        # Plain fixed width byte attributes preserve embedded NUL bytes.
        return value.decode('utf-8')
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _read_and_validate_manifest(h5file):
    meta_group = h5file.get(META_GROUP)
    if meta_group is None or 'manifest' not in meta_group:
        raise ArchiveFormatError(
            "HDF5 file is not a TDMS archive: missing %s group or manifest"
            % META_GROUP)
    format_name = meta_group.attrs.get('format')
    if format_name != ARCHIVE_FORMAT_NAME:
        raise ArchiveFormatError(
            "Unrecognised archive format %r" % format_name)
    version = meta_group.attrs.get('format_version')
    if version != ARCHIVE_FORMAT_VERSION:
        raise ArchiveFormatError(
            "Unsupported archive format version %r; this installation "
            "supports version %d" % (version, ARCHIVE_FORMAT_VERSION))
    try:
        manifest = json.loads(meta_group['manifest'][()])
    except Exception as exc:
        raise ArchiveFormatError("Archive manifest is not valid JSON: %s" % exc)
    if manifest.get('format') != ARCHIVE_FORMAT_NAME:
        raise ArchiveFormatError("Manifest has an unrecognised format name")
    if manifest.get('format_version') != ARCHIVE_FORMAT_VERSION:
        raise ArchiveFormatError(
            "Manifest declares unsupported format version %r"
            % manifest.get('format_version'))
    if manifest.get('tdms_version') not in (4712, 4713):
        raise ArchiveFormatError(
            "Manifest has an invalid TDMS version: %r"
            % manifest.get('tdms_version'))
    return manifest


def _build_restore_records(h5file, manifest):
    """Validate every declared channel and its dataset, returning records."""
    records = []
    for entry in manifest['channels']:
        group_name = entry['group']
        channel_name = entry['name']
        type_enum = entry['type_enum']
        declared_length = entry['length']

        if type_enum != types.Void.enum_value and \
                type_enum not in _SUPPORTED_TYPE_ENUMS:
            raise ArchiveFormatError(
                "Channel /'%s'/'%s' uses an unsupported archived type enum %s"
                % (group_name, channel_name, type_enum))

        try:
            h5_group = h5file[_escape_name(group_name)][_escape_name(channel_name)]
        except KeyError:
            raise ArchiveFormatError(
                "Channel data missing from archive for /'%s'/'%s'"
                % (group_name, channel_name))

        if declared_length == 0 or type_enum == types.Void.enum_value:
            if DATASET_NAME in h5_group:
                dataset = h5_group[DATASET_NAME]
                if dataset.shape != (0,):
                    raise ArchiveFormatError(
                        "Channel /'%s'/'%s' declares zero length but has data"
                        % (group_name, channel_name))
            records.append({
                'group': group_name,
                'name': channel_name,
                'type_enum': type_enum,
                'length': 0,
                'h5_group': h5_group,
                'dataset': None,
            })
            continue

        if DATASET_NAME not in h5_group:
            raise ArchiveFormatError(
                "Channel data missing from archive for /'%s'/'%s'"
                % (group_name, channel_name))
        dataset = h5_group[DATASET_NAME]
        if len(dataset) != declared_length:
            raise ArchiveFormatError(
                "Channel /'%s'/'%s' dataset length %d does not match the "
                "declared length %d"
                % (group_name, channel_name, len(dataset), declared_length))
        _validate_dataset_dtype(dataset, type_enum, group_name, channel_name)

        records.append({
            'group': group_name,
            'name': channel_name,
            'type_enum': type_enum,
            'length': declared_length,
            'h5_group': h5_group,
            'dataset': dataset,
        })
    return records


def _validate_dataset_dtype(dataset, type_enum, group_name, channel_name):
    where = "/'%s'/'%s'" % (group_name, channel_name)
    if type_enum == types.TimeStamp.enum_value:
        if dataset.dtype != _TIMESTAMP_DTYPE:
            raise ArchiveFormatError(
                "Timestamp channel %s has dataset dtype %s, expected %s"
                % (where, dataset.dtype, _TIMESTAMP_DTYPE))
        return
    if type_enum == types.String.enum_value:
        try:
            base = h5py_vlen_base(dataset)
        except Exception as exc:
            raise ArchiveFormatError(
                "String channel %s has an invalid dataset dtype: %s"
                % (where, exc))
        if base != np.dtype('uint8'):
            raise ArchiveFormatError(
                "String channel %s must store uint8 byte arrays" % where)
        return
    expected = _tdms_enum_to_numpy_dtype(type_enum)
    if dataset.dtype != expected:
        raise ArchiveFormatError(
            "Numeric channel %s has dataset dtype %s, expected %s"
            % (where, dataset.dtype, expected))


def h5py_vlen_base(dataset):
    import h5py
    return h5py.check_vlen_dtype(dataset.dtype)


def _write_channel_batches(writer, record, batch_size, nul_group):
    """Restore one channel, feeding bounded batches to the TDMS writer."""
    group_name = record['group']
    channel_name = record['name']
    type_enum = record['type_enum']
    properties = _read_properties(
        record['h5_group'], nul_group,
        _object_key(group_name, channel_name))
    length = record['length']

    if length == 0 or type_enum == types.Void.enum_value:
        # Preserve empty/typeless channels as metadata-only objects.  The
        # writer cannot infer a type for zero length string or timestamp
        # channels, so use a dedicated metadata object with the archived type.
        metadata_type = (
            None if type_enum == types.Void.enum_value
            else _type_for_enum(type_enum))
        writer.write_segment([
            _MetadataOnlyChannel(
                group_name, channel_name, properties,
                data_type=metadata_type)])
        return

    dataset = record['dataset']
    offset = 0
    while offset < length:
        batch = dataset[offset:min(offset + batch_size, length)]
        typed_batch = _batch_to_tdms_data(batch, type_enum)
        writer.write_segment([
            ChannelObject(
                group_name, channel_name, typed_batch, properties,
                data_type=_type_for_enum(type_enum))])
        offset += len(batch)


def _batch_to_tdms_data(batch, type_enum):
    """Convert an archived batch back into typed data for the TDMS writer."""
    if type_enum == types.String.enum_value:
        # Variable length uint8 arrays back to Python strings.
        return np.array(
            [bytes(encoded).decode('utf-8') for encoded in batch], dtype=object)
    if type_enum == types.TimeStamp.enum_value:
        # Wire-order structured array to a TimestampArray, which the writer
        # emits using the exact TDMS timestamp byte layout.
        return TimestampArray(batch)
    if type_enum == types.Boolean.enum_value:
        return batch.astype(np.bool_)
    return np.asarray(batch)


def _type_for_enum(type_enum):
    """Map a TDMS type enum back to its nptdms type class."""
    return types.tds_data_types[type_enum]


class _MetadataOnlyChannel(object):
    """A channel object carrying properties and type metadata but no data.

    The standard :class:`~nptdms.writer.ChannelObject` cannot represent empty
    string or timestamp channels, because it infers the type from sample data.
    This object writes the channel path, explicit type and properties with a
    raw data index marking "no data", matching how empty channels appear in
    real TDMS files.
    """

    def __init__(self, group, channel, properties, data_type=None):
        from nptdms.common import ObjectPath
        self.group = group
        self.channel = channel
        self.properties = properties
        self._path = str(ObjectPath(group, channel))
        self._data_type = data_type

    @property
    def path(self):
        return self._path

    @property
    def has_data(self):
        return False

    @property
    def data_type(self):
        return self._data_type


def _tdms_enum_to_numpy_dtype(type_enum):
    return types.tds_data_types[type_enum].nptype


def _escape_name(name):
    """Map a TDMS name to a reversible HDF5 link name.

    HDF5 uses ``/`` as its path separator and link names cannot be empty,
    while TDMS names have no such restrictions.  Escape ``%``, ``/`` and the
    empty name so every valid TDMS name maps to a unique, valid HDF5 name:

    * ``%`` -> ``%25`` (must be escaped first)
    * ``/`` -> ``%2F``
    * an empty name -> ``%00``
    """
    escaped = name.replace('%', '%25').replace('/', '%2F')
    if escaped == '':
        return '%00'
    return escaped


def _unescape_name(link_name):
    """Reverse :func:`_escape_name`."""
    if link_name == '%00':
        return ''
    return link_name.replace('%2F', '/').replace('%25', '%')


def _require_new_file(path):
    if os.path.exists(path):
        raise FileExistsError(
            "Refusing to overwrite existing file: %s" % path)
