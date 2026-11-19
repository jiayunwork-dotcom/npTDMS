""" Tests for the reversible HDF5 archive format

The archive stores raw, unscaled TDMS data with full type and timestamp
precision so a TDMS file can be rebuilt without the original source.
"""
import json
import os

import h5py
import numpy as np
import pytest

from nptdms import (
    TdmsFile,
    TdmsWriter,
    ChannelObject,
    GroupObject,
    RootObject,
    types,
    export_tdms_to_hdf5,
    restore_tdms_from_hdf5,
    ArchiveFormatError,
    UnsupportedArchiveTypeError,
)
from nptdms.timestamp import TdmsTimestamp
from nptdms.test import scenarios


META_GROUP = ".tdms_archive_meta"


def _build_multi_segment_file(path):
    """Build a multi-segment TDMS file with channels of different lengths."""
    timestamps = [
        TdmsTimestamp(3424234234, 1),
        TdmsTimestamp(-123, 2 ** 64 - 1),
        TdmsTimestamp(0, 0),
        TdmsTimestamp(42, 1 << 33),
    ]
    linear_properties = {
        'NI_Number_Of_Scales': 1,
        'NI_Scale[0]_Scale_Type': 'Linear',
        'NI_Scale[0]_Linear_Slope': 2.0,
        'NI_Scale[0]_Linear_Y_Intercept': 10.0,
    }
    with TdmsWriter(path, version=4713) as writer:
        writer.write_segment([
            RootObject({
                'root_int': np.int16(-4),
                'root_uint': np.uint64(2 ** 63 + 3),
                'root_float': np.float32(1.25),
                'root_double': 2.5,
                'root_bool': True,
                'root_string': 'root value',
                'root_timestamp': TdmsTimestamp(99, 7),
            }),
            GroupObject("measurements", {
                'group_string': "group's value",
                'group_number': 12,
            }),
            ChannelObject(
                "measurements", "temperature",
                np.array([1, 2, 3, 4, 5], dtype=np.int32), linear_properties),
            ChannelObject(
                "measurements", "timestamps", timestamps[:3]),
            ChannelObject(
                "measurements", "text",
                np.array(["", "a\x00b", "café", "\U0001F600"], dtype=object)),
            ChannelObject(
                "measurements", "flags",
                np.array([True, False, True, True], dtype=np.bool_)),
            ChannelObject(
                "measurements", "signed",
                np.array([-8, -16], dtype=np.int8)),
            ChannelObject(
                "measurements", "unsigned",
                np.array([8, 16, 256], dtype=np.uint16)),
            ChannelObject(
                "measurements", "f32",
                np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)),
            ChannelObject(
                "measurements", "f64",
                np.array([1.5, 2.5], dtype=np.float64)),
        ])
        # A second segment appends data to some channels and adds a short one.
        writer.write_segment([
            ChannelObject(
                "measurements", "temperature",
                np.array([6, 7], dtype=np.int32)),
            ChannelObject(
                "measurements", "timestamps", timestamps[3:]),
            ChannelObject(
                "measurements", "short",
                np.array([1234], dtype=np.int16)),
            GroupObject("empty_group", {'empty_group_prop': False}),
            ChannelObject(
                "measurements", "empty_int",
                np.array([], dtype=np.int32)),
            ChannelObject(
                "measurements", "empty_string",
                np.array([], dtype=object), data_type=types.String),
        ])


def _assert_files_equivalent(source_path, restored_path):
    source = TdmsFile.read(source_path, raw_timestamps=True)
    restored = TdmsFile.read(restored_path, raw_timestamps=True)

    assert restored.tdms_version == source.tdms_version
    assert [g.name for g in restored.groups()] == \
        [g.name for g in source.groups()]
    _assert_properties_equal(source.properties, restored.properties)

    for source_group in source.groups():
        restored_group = restored[source_group.name]
        _assert_properties_equal(
            source_group.properties, restored_group.properties)
        assert [c.name for c in source_group.channels()] == \
            [c.name for c in restored_group.channels()]
        for source_channel in source_group.channels():
            restored_channel = restored_group[source_channel.name]
            _assert_channel_equal(source_channel, restored_channel)


def _assert_channel_equal(source_channel, restored_channel):
    assert len(restored_channel) == len(source_channel)
    assert restored_channel.data_type is source_channel.data_type
    assert restored_channel._raw_data_dtype() == source_channel._raw_data_dtype()
    _assert_properties_equal(
        source_channel.properties, restored_channel.properties)

    source_raw = source_channel.read_data(scaled=False)
    restored_raw = restored_channel.read_data(scaled=False)
    if source_channel.data_type is types.TimeStamp:
        assert list(restored_raw) == list(source_raw)
    elif source_channel.data_type is types.String:
        assert list(restored_raw) == list(source_raw)
    else:
        np.testing.assert_array_equal(restored_raw, source_raw)

    # Scaled reads must also match the source.
    if len(source_channel) > 0:
        source_scaled = source_channel.read_data(scaled=True)
        restored_scaled = restored_channel.read_data(scaled=True)
        if source_channel.data_type is types.String:
            assert list(restored_scaled) == list(source_scaled)
        elif source_channel.data_type is types.TimeStamp:
            assert list(restored_scaled) == list(source_scaled)
        else:
            np.testing.assert_array_equal(restored_scaled, source_scaled)


def _assert_properties_equal(expected, actual):
    assert list(actual.keys()) == list(expected.keys())
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, TdmsTimestamp):
            assert isinstance(actual_value, TdmsTimestamp)
            assert actual_value == expected_value
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value
            if isinstance(expected_value, np.number):
                assert np.dtype(type(actual_value)) == np.dtype(type(expected_value))


def test_archive_round_trip_with_two_batch_sizes(tmp_path):
    """End to end archiving and restore, moving the source away in between."""
    source_path = tmp_path / "source.tdms"
    archive_path = tmp_path / "archive.h5"
    _build_multi_segment_file(str(source_path))

    source = TdmsFile.read(str(source_path), raw_timestamps=True)
    export_tdms_to_hdf5(source, str(archive_path))

    # The archive must be readable and sliceable on its own.
    with h5py.File(archive_path, "r") as archive:
        meta = archive[META_GROUP]
        assert meta.attrs["format"] == "nptdms-tdms-archive"
        assert meta.attrs["format_version"] == 1
        assert meta.attrs["tdms_version"] == 4713
        manifest = json.loads(meta["manifest"][()])
        assert manifest["format_version"] == 1
        temperature = archive["measurements"]["temperature"]["data"]
        assert temperature.dtype == np.dtype("int32")
        np.testing.assert_array_equal(temperature[2:5], [3, 4, 5])
        text = archive["measurements"]["text"]["data"]
        assert bytes(text[1]).decode("utf-8") == "a\x00b"
        timestamps = archive["measurements"]["timestamps"]["data"]
        assert timestamps.dtype.names == ("second_fractions", "seconds")
        np.testing.assert_array_equal(
            timestamps[1]["seconds"], np.int64(-123))

    # Restore with the source removed, using two different batch sizes.
    moved_source = tmp_path / "source_moved.tdms"
    os.replace(source_path, moved_source)
    for batch_size in (2, 100000):
        restored_path = tmp_path / ("restored_%d.tdms" % batch_size)
        restore_tdms_from_hdf5(
            str(archive_path), str(restored_path), batch_size=batch_size)
        _assert_files_equivalent(str(moved_source), str(restored_path))


def test_archive_loaded_and_streamed_exports_match(tmp_path):
    """Exporting a fully read file and an opened file give identical archives."""
    source_path = tmp_path / "source.tdms"
    _build_multi_segment_file(str(source_path))

    loaded_archive = tmp_path / "loaded.h5"
    loaded = TdmsFile.read(str(source_path), raw_timestamps=True)
    export_tdms_to_hdf5(loaded, str(loaded_archive))

    streamed_archive = tmp_path / "streamed.h5"
    with TdmsFile.open(str(source_path), raw_timestamps=True) as opened:
        export_tdms_to_hdf5(opened, str(streamed_archive))

    with h5py.File(loaded_archive) as first, h5py.File(streamed_archive) as second:
        def compare(name, obj):
            other = second[name]
            if isinstance(obj, h5py.Dataset):
                if obj.shape == ():
                    assert obj[()] == other[()]
                elif obj.dtype.kind == "O" or h5py.check_vlen_dtype(obj.dtype) is not None:
                    assert [bytes(v) for v in obj[:]] == \
                        [bytes(v) for v in other[:]]
                else:
                    np.testing.assert_array_equal(obj[:], other[:])
        first.visititems(compare)


def test_timestamp_fractions_keep_full_64bit_resolution(tmp_path):
    """Two timestamps differing by a single fraction bit stay distinct."""
    source_path = tmp_path / "timestamps.tdms"
    timestamps = [
        TdmsTimestamp(1000, 123456789),
        TdmsTimestamp(1000, 123456790),
        TdmsTimestamp(-1, 2 ** 64 - 1),
    ]
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "ts", timestamps)])

    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(
        TdmsFile.read(str(source_path), raw_timestamps=True), str(archive_path))
    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path)

    restored = TdmsFile.read(restored_path, raw_timestamps=True)
    values = list(restored["g"]["ts"][:])
    assert values == timestamps
    assert values[0].second_fractions + 1 == values[1].second_fractions
    assert values[2].seconds == -1
    assert values[2].second_fractions == 2 ** 64 - 1


def test_string_edge_cases_round_trip(tmp_path):
    """Empty strings, non-ASCII text and embedded NUL bytes are preserved."""
    source_path = tmp_path / "strings.tdms"
    strings = ["", "a\x00b", "\x00leading", "trailing\x00",
               "café", "\U0001F600", "plain"]
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "s", np.array(strings, dtype=object))])

    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)
    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path)

    restored = TdmsFile.read(restored_path)
    assert list(restored["g"]["s"][:]) == strings


def test_nul_bytes_in_properties_round_trip(tmp_path):
    """String properties with embedded or trailing NUL bytes are preserved."""
    source_path = tmp_path / "properties.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            RootObject({"root_nul": "x\x00y"}),
            GroupObject("g", {"only_nuls": "\x00\x00", "plain": "ok"}),
            ChannelObject(
                "g", "c", np.arange(2, dtype=np.int32),
                {"trailing_nul": "z\x00"}),
        ])

    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)
    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path)

    restored = TdmsFile.read(restored_path)
    assert restored.properties["root_nul"] == "x\x00y"
    assert restored["g"].properties["only_nuls"] == "\x00\x00"
    assert restored["g"].properties["plain"] == "ok"
    assert restored["g"]["c"].properties["trailing_nul"] == "z\x00"


def test_special_group_and_channel_names(tmp_path):
    """Names with slashes, quotes and spaces round trip without splitting."""
    source_path = tmp_path / "names.tdms"
    group_name = "g/ 'quoted' space"
    channel_name = 'c/"nested"/end'
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            GroupObject(group_name, {"prop/with/slash": 1}),
            ChannelObject(
                group_name, channel_name,
                np.arange(3, dtype=np.int32),
                {"channel prop": "x/y"}),
        ])

    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)
    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path)

    restored = TdmsFile.read(restored_path)
    assert group_name in restored
    assert channel_name in restored[group_name]
    assert restored[group_name].properties["prop/with/slash"] == 1
    assert restored[group_name][channel_name].properties["channel prop"] == "x/y"
    np.testing.assert_array_equal(
        restored[group_name][channel_name][:], [0, 1, 2])


def test_empty_groups_and_channels_are_preserved(tmp_path):
    source_path = tmp_path / "empty.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            GroupObject("empty_group", {"prop": 3}),
            ChannelObject("data_group", "values",
                          np.arange(4, dtype=np.int32)),
            ChannelObject("data_group", "empty_int",
                          np.array([], dtype=np.int32)),
            ChannelObject("data_group", "empty_string",
                          np.array([], dtype=object), data_type=types.String),
        ])

    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)
    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path)

    restored = TdmsFile.read(restored_path)
    assert "empty_group" in restored
    assert restored["empty_group"].channels() == []
    assert restored["empty_group"].properties["prop"] == 3
    group = restored["data_group"]
    assert len(group["values"]) == 4
    assert len(group["empty_int"]) == 0
    assert group["empty_int"].data_type is types.Int32
    assert len(group["empty_string"]) == 0
    assert group["empty_string"].data_type is types.String


def test_unsupported_types_are_rejected(tmp_path):
    archive_path = tmp_path / "archive.h5"
    test_file, _ = scenarios.complex_data().values
    tdms_file = test_file.load()
    with pytest.raises(UnsupportedArchiveTypeError) as exc_info:
        export_tdms_to_hdf5(tdms_file, archive_path)
    assert "complex_single_channel" in str(exc_info.value)
    assert not archive_path.exists()


def test_truncated_source_is_rejected(tmp_path):
    test_file, _ = scenarios.incomplete_last_segment().values
    archive_path = tmp_path / "archive.h5"
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(Exception):
                export_tdms_to_hdf5(tdms_file, archive_path)
    assert not archive_path.exists()


def test_physically_truncated_source_is_rejected(tmp_path):
    """Cutting a valid file at several points must always be detected."""
    test_file, _ = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        contents = temp_file.file.read()

    for fraction in (0.5, 0.6, 0.75, 0.9):
        cut = int(len(contents) * fraction)
        source_path = tmp_path / ("cut_%s.tdms" % str(fraction).replace('.', '_'))
        source_path.write_bytes(contents[:cut])
        archive_path = tmp_path / ("cut_%s.h5" % str(fraction).replace('.', '_'))
        with TdmsFile.open(source_path) as tdms_file:
            with pytest.raises(Exception):
                export_tdms_to_hdf5(tdms_file, archive_path)
        assert not archive_path.exists()


def test_target_paths_must_be_new_files(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    with pytest.raises(FileExistsError):
        export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    restored_path = tmp_path / "restored.tdms"
    restored_path.write_text("existing")
    with pytest.raises(FileExistsError):
        restore_tdms_from_hdf5(archive_path, restored_path)
    # The existing target must be left untouched.
    assert restored_path.read_text() == "existing"


def test_unknown_archive_version_is_rejected(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    with h5py.File(archive_path, "a") as archive:
        archive[META_GROUP].attrs["format_version"] = 999
        manifest = json.loads(archive[META_GROUP]["manifest"][()])
        manifest["format_version"] = 999
        del archive[META_GROUP]["manifest"]
        archive[META_GROUP].create_dataset(
            "manifest", shape=(), dtype=h5py.string_dtype(),
            data=json.dumps(manifest))

    restored_path = tmp_path / "restored.tdms"
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(archive_path, restored_path)
    assert not restored_path.exists()


def test_missing_channel_data_is_rejected(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    with h5py.File(archive_path, "a") as archive:
        del archive["g"]["a"]["data"]

    restored_path = tmp_path / "restored.tdms"
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(archive_path, restored_path)
    assert not restored_path.exists()


def test_length_mismatch_is_rejected(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    with h5py.File(archive_path, "a") as archive:
        manifest = json.loads(archive[META_GROUP]["manifest"][()])
        manifest["channels"][0]["length"] = 99
        del archive[META_GROUP]["manifest"]
        archive[META_GROUP].create_dataset(
            "manifest", shape=(), dtype=h5py.string_dtype(),
            data=json.dumps(manifest))

    restored_path = tmp_path / "restored.tdms"
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(archive_path, restored_path)
    assert not restored_path.exists()


def test_dtype_mismatch_is_rejected(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    with h5py.File(archive_path, "a") as archive:
        del archive["g"]["a"]["data"]
        archive["g"]["a"].create_dataset(
            "data", data=np.arange(3, dtype=np.int64))

    restored_path = tmp_path / "restored.tdms"
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(archive_path, restored_path)
    assert not restored_path.exists()


def test_non_archive_hdf5_file_is_rejected(tmp_path):
    plain_path = tmp_path / "plain.h5"
    with h5py.File(plain_path, "w") as plain:
        plain.create_dataset("data", data=np.arange(4))
    restored_path = tmp_path / "restored.tdms"
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(plain_path, restored_path)
    assert not restored_path.exists()


def test_invalid_batch_size_is_rejected(tmp_path):
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)
    with pytest.raises(ValueError):
        restore_tdms_from_hdf5(
            archive_path, tmp_path / "restored.tdms", batch_size=0)


def test_closed_reader_and_index_only_files_are_rejected(tmp_path):
    from nptdms.export.hdf_archive import ArchiveError

    source_path = tmp_path / "source.tdms"
    with TdmsWriter(str(source_path), index_file=True) as writer:
        writer.write_segment([
            ChannelObject("g", "a", np.arange(3, dtype=np.int32))])

    # An opened file whose reader has been closed.
    with TdmsFile.open(source_path) as tdms_file:
        pass
    with pytest.raises(ArchiveError):
        export_tdms_to_hdf5(tdms_file, tmp_path / "closed.h5")

    # An index file without its data file has no raw samples.
    index_file = TdmsFile.read(str(source_path) + "_index")
    with pytest.raises(ArchiveError):
        export_tdms_to_hdf5(index_file, tmp_path / "index_only.h5")


def test_restore_buffers_are_bounded(tmp_path):
    """Restore must never materialise a whole long channel in one batch."""
    source_path = tmp_path / "source.tdms"
    length = 10000
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "long",
                          np.arange(length, dtype=np.int32))])
    archive_path = tmp_path / "archive.h5"
    export_tdms_to_hdf5(TdmsFile.read(source_path), archive_path)

    restored_path = tmp_path / "restored.tdms"
    restore_tdms_from_hdf5(archive_path, restored_path, batch_size=137)
    restored = TdmsFile.read(restored_path)
    np.testing.assert_array_equal(
        restored["g"]["long"][:], np.arange(length))


def test_export_buffers_are_bounded_when_streaming(tmp_path):
    """Streaming export must process source chunks without full channel loads."""
    source_path = tmp_path / "source.tdms"
    length = 5000
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            ChannelObject("g", "long",
                          np.arange(length, dtype=np.int64))])
    archive_path = tmp_path / "archive.h5"
    with TdmsFile.open(source_path) as tdms_file:
        assert not tdms_file.data_read
        export_tdms_to_hdf5(tdms_file, archive_path)
        assert not tdms_file.data_read

    with h5py.File(archive_path) as archive:
        np.testing.assert_array_equal(
            archive["g"]["long"]["data"][:], np.arange(length))


def test_daqmx_channels_are_rejected(tmp_path):
    from nptdms.test.test_daqmx import (
        daqmx_scaler_metadata, daqmx_channel_metadata, segment_toc)
    from nptdms.test.util import (
        GeneratedFile, segment_objects_metadata,
        root_metadata, group_metadata)

    scaler_metadata = daqmx_scaler_metadata(0, 3, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, "01 00 02 00")
    tdms_file = test_file.load()

    archive_path = tmp_path / "archive.h5"
    with pytest.raises(UnsupportedArchiveTypeError) as exc_info:
        export_tdms_to_hdf5(tdms_file, archive_path)
    assert "Channel1" in str(exc_info.value)
    assert "DaqMx" in str(exc_info.value)
    assert not archive_path.exists()


def test_legacy_as_hdf_layout_is_unchanged(tmp_path):
    """Regression check: the original as_hdf export keeps its old layout."""
    source_path = tmp_path / "source.tdms"
    with TdmsWriter(source_path) as writer:
        writer.write_segment([
            RootObject({"root_prop": 11}),
            GroupObject("group", {"group_prop": "value"}),
            ChannelObject(
                "group", "integers",
                np.array([1, 2, 3, 4], dtype=np.int32),
                {"wf_increment": 0.1}),
            ChannelObject(
                "group", "text",
                np.array(["abc", "def"], dtype=object)),
        ])

    legacy_path = tmp_path / "legacy.h5"
    tdms_file = TdmsFile.read(source_path)
    h5_file = tdms_file.as_hdf(legacy_path)
    try:
        # The legacy layout maps groups and channels directly to HDF5 groups
        # and datasets with no extra archive metadata.
        assert META_GROUP not in h5_file
        integers = h5_file["group"]["integers"]
        assert integers.dtype == np.dtype("int32")
        np.testing.assert_array_equal(integers[...], [1, 2, 3, 4])
        assert integers.attrs["wf_increment"] == 0.1
        assert h5_file["group"].attrs["group_prop"] == "value"
        assert h5_file.attrs["root_prop"] == 11
        text = h5_file["group"]["text"]
        assert text.dtype.kind == "O"
        assert [value.decode("utf-8") if isinstance(value, bytes) else value
                for value in text[...]] == ["abc", "def"]
    finally:
        h5_file.close()

    # The legacy and archive files are distinct formats: the legacy file must
    # not be accepted as an archive.
    with pytest.raises(ArchiveFormatError):
        restore_tdms_from_hdf5(
            legacy_path, tmp_path / "restored.tdms")
