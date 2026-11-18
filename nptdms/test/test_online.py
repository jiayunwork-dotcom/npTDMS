"""Tests for online reading of TDMS files that are still being written"""

import os
import tempfile

import numpy as np
import pytest

from nptdms import TdmsFile, OnlineTdmsError
from nptdms.test.util import (
    GeneratedFile, segment_objects_metadata, root_metadata, group_metadata,
    channel_metadata, channel_metadata_with_repeated_structure,
    hex_properties, hexlify_value)


TOC = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
INT32 = 3
DOUBLE = 10


def build_segment(toc, metadata, data, incomplete=False):
    test_file = GeneratedFile()
    test_file.add_segment(toc, metadata, data, incomplete=incomplete)
    return test_file._get_contents()


def build_index(*segments):
    test_file = GeneratedFile()
    for (toc, metadata, data) in segments:
        test_file.add_segment(toc, metadata, data)
    return test_file._get_index_contents()


def int32s(*values):
    return " ".join(hexlify_value("<i", value) for value in values)


def append_bytes(path, data):
    with open(path, "ab") as f:
        f.write(data)


def write_bytes(path, offset, data):
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)


def collect_file_chunks(tdms_file):
    chunks = []
    for chunk in tdms_file.data_chunks():
        chunk_data = {}
        for group in chunk.groups():
            for channel in group.channels():
                chunk_data["%s/%s" % (group.name, channel.name)] = (channel.offset, channel[:])
        chunks.append(chunk_data)
    return chunks


def collect_channel_chunks(channel):
    return [(chunk.offset, chunk[:]) for chunk in channel.data_chunks()]


def assert_view_matches_reference(view, file_bytes):
    """Compare an online view against the offline reading of the published prefix"""
    fd, path = tempfile.mkstemp(suffix=".tdms")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        with TdmsFile.open(path) as reference:
            assert view.properties == reference.properties
            assert [g.name for g in view.groups()] == [g.name for g in reference.groups()]
            for ref_group in reference.groups():
                view_group = view[ref_group.name]
                assert view_group.properties == ref_group.properties
                assert [c.name for c in view_group.channels()] == [c.name for c in ref_group.channels()]
                for ref_channel in ref_group.channels():
                    view_channel = view_group[ref_channel.name]
                    assert len(view_channel) == len(ref_channel)
                    assert view_channel.properties == ref_channel.properties
                    assert view_channel.data_type == ref_channel.data_type
                    np.testing.assert_array_equal(view_channel[:], ref_channel[:])
                    np.testing.assert_array_equal(view_channel.read_data(), ref_channel.read_data())
                    if len(ref_channel) > 1:
                        np.testing.assert_array_equal(view_channel[1:], ref_channel[1:])
                        np.testing.assert_array_equal(view_channel[:-1], ref_channel[:-1])
            # File level chunks with channel offsets
            view_chunks = collect_file_chunks(view)
            ref_chunks = collect_file_chunks(reference)
            assert len(view_chunks) == len(ref_chunks)
            for view_chunk, ref_chunk in zip(view_chunks, ref_chunks):
                assert view_chunk.keys() == ref_chunk.keys()
                for name, (offset, data) in ref_chunk.items():
                    assert view_chunk[name][0] == offset
                    np.testing.assert_array_equal(view_chunk[name][1], data)
            # Channel level chunks with offsets
            for ref_group in reference.groups():
                for ref_channel in ref_group.channels():
                    view_channel = view[ref_group.name][ref_channel.name]
                    view_chunks = collect_channel_chunks(view_channel)
                    ref_chunks = collect_channel_chunks(ref_channel)
                    assert len(view_chunks) == len(ref_chunks)
                    for (view_offset, view_data), (ref_offset, ref_data) in zip(view_chunks, ref_chunks):
                        assert view_offset == ref_offset
                        np.testing.assert_array_equal(view_data, ref_data)
    finally:
        os.unlink(path)


def segment_one():
    """Root, group and one int32 channel with values [1, 2]"""
    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(),
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    return build_segment(TOC, metadata, int32s(1, 2))


def test_segment_written_in_pieces(tmp_path):
    """A segment split over several writes is only published once complete"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()
    seg2_metadata = segment_objects_metadata(
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    seg2 = build_segment(TOC, seg2_metadata, int32s(3, 4))

    with TdmsFile.open_online(path) as online_file:
        # File starts at zero bytes
        view = online_file.refresh()
        assert view.groups() == []

        # Partial lead-in
        append_bytes(path, seg1[:10])
        assert online_file.refresh().groups() == []

        # Rest of the lead-in, then part of the metadata
        append_bytes(path, seg1[10:40])
        assert online_file.refresh().groups() == []

        # Remaining metadata and part of the payload
        append_bytes(path, seg1[40:-4])
        assert online_file.refresh().groups() == []

        # Final payload bytes complete the segment
        append_bytes(path, seg1[-4:])
        view = online_file.refresh()
        channel = view["Group"]["Channel1"]
        np.testing.assert_array_equal(channel[:], [1, 2])
        assert_view_matches_reference(view, seg1)

        # Second segment also arrives in pieces
        append_bytes(path, seg2[:16])
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2
        append_bytes(path, seg2[16:])
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4])
        assert_view_matches_reference(view, seg1 + seg2)


def test_backfilled_segment_end_position(tmp_path):
    """A segment with unknown end position is published once the writer backfills it"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()
    seg2_metadata = segment_objects_metadata(
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    seg2_incomplete = build_segment(TOC, seg2_metadata, int32s(3, 4), incomplete=True)
    seg2_complete = build_segment(TOC, seg2_metadata, int32s(3, 4))
    assert len(seg2_incomplete) == len(seg2_complete)
    # The two versions only differ in the next segment offset field of the lead-in
    assert seg2_incomplete[12:20] == b'\xFF' * 8
    assert seg2_incomplete[:12] == seg2_complete[:12]
    assert seg2_incomplete[20:] == seg2_complete[20:]

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2

        # The whole segment is present but its end position is unknown
        append_bytes(path, seg2_incomplete)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2

        # The writer backfills the end position of the segment
        write_bytes(path, len(seg1) + 12, seg2_complete[12:20])
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4])
        assert_view_matches_reference(view, seg1 + seg2_complete)


def test_interleaved_file_written_in_pieces(tmp_path):
    """Interleaved data is readable once the segment is complete"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    toc = TOC + ("kTocInterleavedData",)
    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(),
        channel_metadata("/'Group'/'Channel1'", INT32, 2),
        channel_metadata("/'Group'/'Channel2'", INT32, 2))
    # Interleaved: ch1[0], ch2[0], ch1[1], ch2[1]
    seg = build_segment(toc, metadata, int32s(1, 10, 2, 20))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg[:-6])
        assert online_file.refresh().groups() == []
        append_bytes(path, seg[-6:])
        view = online_file.refresh()
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2])
        np.testing.assert_array_equal(view["Group"]["Channel2"][:], [10, 20])
        assert_view_matches_reference(view, seg)


def test_index_file_arriving_before_data_file(tmp_path):
    """An index that is written early must not publish segments the data file doesn't have"""
    path = str(tmp_path / "test.tdms")
    index_path = path + "_index"
    open(path, "wb").close()

    seg1 = segment_one()
    seg2_metadata = segment_objects_metadata(
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    seg2 = build_segment(TOC, seg2_metadata, int32s(3, 4))

    # The index already describes both segments while the data file only has the first
    index_contents = build_index(
        (TOC, segment_objects_metadata(
            root_metadata(), group_metadata(),
            channel_metadata("/'Group'/'Channel1'", INT32, 2)), int32s(1, 2)),
        (TOC, seg2_metadata, int32s(3, 4)))
    with open(index_path, "wb") as f:
        f.write(index_contents)

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2])

        append_bytes(path, seg2)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        assert_view_matches_reference(view, seg1 + seg2)


def test_index_file_arriving_late(tmp_path):
    """A missing or stale index must not drop segments that are complete in the data file"""
    path = str(tmp_path / "test.tdms")
    index_path = path + "_index"
    open(path, "wb").close()

    seg1 = segment_one()
    seg2_metadata = segment_objects_metadata(
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    seg2 = build_segment(TOC, seg2_metadata, int32s(3, 4))

    with TdmsFile.open_online(path) as online_file:
        # No index file at all yet
        append_bytes(path, seg1 + seg2)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        assert_view_matches_reference(view, seg1 + seg2)

        # A stale index that only covers the first segment arrives late
        index_contents = build_index(
            (TOC, segment_objects_metadata(
                root_metadata(), group_metadata(),
                channel_metadata("/'Group'/'Channel1'", INT32, 2)), int32s(1, 2)))
        with open(index_path, "wb") as f:
            f.write(index_contents)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4])
        assert_view_matches_reference(view, seg1 + seg2)


def test_segments_reusing_previous_layout(tmp_path):
    """Segments that reuse the previous channel layout are picked up by refresh"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(),
        channel_metadata("/'Group'/'Channel1'", INT32, 2),
        channel_metadata("/'Group'/'Channel2'", INT32, 2))
    seg1 = build_segment(TOC, metadata, int32s(1, 2, 10, 20))
    # Segment without any metadata reuses the previous segment layout
    seg2 = build_segment(("kTocRawData",), "", int32s(3, 4, 30, 40))
    # Segment with metadata that refers back to the previous raw data index
    seg3_metadata = segment_objects_metadata(
        channel_metadata_with_repeated_structure("/'Group'/'Channel1'"),
        channel_metadata_with_repeated_structure("/'Group'/'Channel2'"))
    seg3 = build_segment(
        ("kTocMetaData", "kTocRawData"), seg3_metadata, int32s(5, 6, 50, 60))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2

        append_bytes(path, seg2)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4
        assert len(view["Group"]["Channel2"]) == 4
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4])
        np.testing.assert_array_equal(view["Group"]["Channel2"][:], [10, 20, 30, 40])
        assert_view_matches_reference(view, seg1 + seg2)

        append_bytes(path, seg3)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 6
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(view["Group"]["Channel2"][:], [10, 20, 30, 40, 50, 60])
        assert_view_matches_reference(view, seg1 + seg2 + seg3)


def test_unequal_length_channels_and_chunk_offsets(tmp_path):
    """Channels with different lengths per chunk get correct cumulative offsets"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(),
        channel_metadata("/'Group'/'Long'", INT32, 4),
        channel_metadata("/'Group'/'Short'", INT32, 2))
    # Two chunks: long channel data then short channel data per chunk
    seg1 = build_segment(TOC, metadata, int32s(1, 2, 3, 4, 101, 102, 5, 6, 7, 8, 103, 104))
    seg2 = build_segment(("kTocRawData",), "", int32s(9, 10, 11, 12, 105, 106))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        view = online_file.refresh()
        long_channel = view["Group"]["Long"]
        short_channel = view["Group"]["Short"]
        assert len(long_channel) == 8
        assert len(short_channel) == 4
        long_chunks = collect_channel_chunks(long_channel)
        short_chunks = collect_channel_chunks(short_channel)
        assert [offset for offset, _ in long_chunks] == [0, 4]
        assert [offset for offset, _ in short_chunks] == [0, 2]

        append_bytes(path, seg2)
        view = online_file.refresh()
        long_chunks = collect_channel_chunks(view["Group"]["Long"])
        short_chunks = collect_channel_chunks(view["Group"]["Short"])
        assert [offset for offset, _ in long_chunks] == [0, 4, 8]
        assert [offset for offset, _ in short_chunks] == [0, 2, 4]
        np.testing.assert_array_equal(view["Group"]["Long"][:], list(range(1, 13)))
        np.testing.assert_array_equal(view["Group"]["Short"][:], [101, 102, 103, 104, 105, 106])
        assert_view_matches_reference(view, seg1 + seg2)


def test_corrupt_segment_fails_refresh_and_old_view_survives(tmp_path):
    """A boundary-complete but corrupt segment fails the refresh without a partial view"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()
    # Segment with a complete boundary but an unrecognised data type in its metadata
    corrupt_metadata = segment_objects_metadata(
        channel_metadata("/'Group'/'Channel1'", 0x0000FFFF, 2))
    corrupt_seg = build_segment(TOC, corrupt_metadata, int32s(3, 4))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        old_view = online_file.refresh()
        old_channel = old_view["Group"]["Channel1"]
        np.testing.assert_array_equal(old_channel[:], [1, 2])

        append_bytes(path, corrupt_seg)
        with pytest.raises(OnlineTdmsError):
            online_file.refresh()
        # The corruption is persistent, refreshing again keeps failing
        with pytest.raises(OnlineTdmsError):
            online_file.refresh()

        # The old view of the unchanged prefix is still fully readable
        assert len(old_channel) == 2
        np.testing.assert_array_equal(old_channel[:], [1, 2])
        np.testing.assert_array_equal(old_channel.read_data(), [1, 2])
        chunks = collect_channel_chunks(old_channel)
        assert [offset for offset, _ in chunks] == [0]
        np.testing.assert_array_equal(chunks[0][1], [1, 2])
        assert_view_matches_reference(old_view, seg1)


def test_truncated_file_is_rejected(tmp_path):
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()
    seg2 = build_segment(
        TOC, segment_objects_metadata(channel_metadata("/'Group'/'Channel1'", INT32, 2)),
        int32s(3, 4))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1 + seg2)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 4

        os.truncate(path, len(seg1) + 10)
        with pytest.raises(OnlineTdmsError, match="truncated"):
            online_file.refresh()


def test_replaced_file_is_rejected(tmp_path):
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        view = online_file.refresh()
        assert len(view["Group"]["Channel1"]) == 2

        # Replace the file with a different valid TDMS file
        replacement = str(tmp_path / "replacement.tdms")
        with open(replacement, "wb") as f:
            f.write(seg1 + seg1)
        os.replace(replacement, path)
        with pytest.raises(OnlineTdmsError, match="replaced"):
            online_file.refresh()


def test_view_snapshot_isolation(tmp_path):
    """Channels and chunk iterators from an old view keep reading the old view"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    # First segment has two chunks of two values
    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(),
        channel_metadata("/'Group'/'Channel1'", INT32, 2))
    seg1 = build_segment(TOC, metadata, int32s(1, 2, 3, 4))
    # Second segment adds new file and channel properties
    root_with_property = (
        "01 00 00 00"  # Length of the object path
        "2F"           # Object path (/)
        "FF FF FF FF"  # Raw data index
        + hex_properties({"new_file_prop": (INT32, hexlify_value("<i", 42))}))
    seg2_metadata = segment_objects_metadata(
        root_with_property,
        channel_metadata("/'Group'/'Channel1'", INT32, 2,
                         {"wf_increment": (DOUBLE, hexlify_value("<d", 0.1))}))
    seg2 = build_segment(("kTocMetaData", "kTocRawData"), seg2_metadata, int32s(5, 6, 7, 8))

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        old_view = online_file.refresh()
        old_channel = old_view["Group"]["Channel1"]
        assert len(old_channel) == 4
        chunk_iterator = old_channel.data_chunks()
        first_chunk = next(chunk_iterator)
        assert first_chunk.offset == 0
        np.testing.assert_array_equal(first_chunk[:], [1, 2])

        append_bytes(path, seg2)
        new_view = online_file.refresh()

        # The new view sees the new segment and the new properties
        new_channel = new_view["Group"]["Channel1"]
        assert len(new_channel) == 8
        assert new_view.properties["new_file_prop"] == 42
        assert new_channel.properties["wf_increment"] == 0.1
        np.testing.assert_array_equal(new_channel[:], [1, 2, 3, 4, 5, 6, 7, 8])
        assert_view_matches_reference(new_view, seg1 + seg2)

        # The old view is unchanged
        assert "new_file_prop" not in old_view.properties
        assert "wf_increment" not in old_channel.properties
        assert len(old_channel) == 4
        np.testing.assert_array_equal(old_channel[:], [1, 2, 3, 4])

        # The iterator started before the refresh keeps reading the old view
        second_chunk = next(chunk_iterator)
        assert second_chunk.offset == 2
        np.testing.assert_array_equal(second_chunk[:], [3, 4])
        with pytest.raises(StopIteration):
            next(chunk_iterator)

        # File level chunks of the old view only cover the first segment
        old_file_chunks = collect_file_chunks(old_view)
        assert len(old_file_chunks) == 2
        new_file_chunks = collect_file_chunks(new_view)
        assert len(new_file_chunks) == 4


def test_repeated_refresh_without_new_data(tmp_path):
    """Refreshing without new complete segments does not duplicate channel data"""
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()

    with TdmsFile.open_online(path) as online_file:
        append_bytes(path, seg1)
        first_view = online_file.refresh()
        second_view = online_file.refresh()
        assert len(first_view["Group"]["Channel1"]) == 2
        assert len(second_view["Group"]["Channel1"]) == 2
        np.testing.assert_array_equal(second_view["Group"]["Channel1"][:], [1, 2])
        assert len(collect_file_chunks(second_view)) == 1


def test_close_semantics(tmp_path):
    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    seg1 = segment_one()
    seg2 = build_segment(
        TOC, segment_objects_metadata(channel_metadata("/'Group'/'Channel1'", INT32, 2)),
        int32s(3, 4))

    online_file = TdmsFile.open_online(path)
    append_bytes(path, seg1)
    view = online_file.refresh()
    channel = view["Group"]["Channel1"]

    # Closing the online reader does not close views it already delivered
    append_bytes(path, seg2)
    online_file.close()
    with pytest.raises(RuntimeError):
        online_file.refresh()
    np.testing.assert_array_equal(channel[:], [1, 2])

    # Closing a view releases its file handle
    view.close()
    with pytest.raises(RuntimeError):
        channel.read_data()


def test_open_online_rejects_file_object(tmp_path):
    path = tmp_path / "test.tdms"
    path.write_bytes(segment_one())
    with open(path, "rb") as f:
        with pytest.raises(ValueError):
            TdmsFile.open_online(f)


def test_open_online_accepts_pathlib_path(tmp_path):
    path = tmp_path / "test.tdms"
    path.write_bytes(segment_one())
    with TdmsFile.open_online(path) as online_file:
        view = online_file.refresh()
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2])


@pytest.mark.parametrize("index_file", [False, True])
def test_segments_appended_by_tdms_writer(tmp_path, index_file):
    """Segments appended by TdmsWriter are picked up by each refresh"""
    from nptdms import TdmsWriter, ChannelObject

    path = str(tmp_path / "test.tdms")
    open(path, "wb").close()

    with TdmsFile.open_online(path) as online_file:
        assert online_file.refresh().groups() == []
        with TdmsWriter(path, mode='a', index_file=index_file) as writer:
            writer.write_segment([
                ChannelObject("Group", "Channel1", np.array([1, 2], dtype=np.int32))])
        view = online_file.refresh()
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2])
        with TdmsWriter(path, mode='a', index_file=index_file) as writer:
            writer.write_segment([
                ChannelObject("Group", "Channel1", np.array([3, 4], dtype=np.int32))])
        view = online_file.refresh()
        np.testing.assert_array_equal(view["Group"]["Channel1"][:], [1, 2, 3, 4])
