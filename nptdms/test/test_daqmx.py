"""Test reading of TDMS files with DAQmx data
"""

from collections import defaultdict
import logging
import tempfile
import numpy as np
import pytest

from nptdms import TdmsFile
from nptdms import types
from nptdms.log import log_manager
from nptdms.test.scenarios import timestamp_hexlify
from nptdms.test.util import (
    GeneratedFile, hexlify_value, string_hexlify, segment_objects_metadata, hex_properties,
    root_metadata, group_metadata)


def test_single_channel_i16():
    """ Test loading a DAQmx file with a single channel of I16 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 3, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.int16
    np.testing.assert_array_equal(data, [1, 2, -1, -2])


def test_single_channel_u16():
    """ Test loading a DAQmx file with a single channel of U16 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, [1, 2, 2**16 - 1, 2**16 - 2])


def test_single_channel_i32():
    """ Test loading a DAQmx file with a single channel of I32 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 5, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "FF FF FF FF"
        "FE FF FF FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.int32
    np.testing.assert_array_equal(data, [1, 2, -1, -2])


def test_single_channel_u32():
    """ Test loading a DAQmx file with a single channel of U32 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 4, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "FF FF FF FF"
        "FE FF FF FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint32
    np.testing.assert_array_equal(data, [1, 2, 2**32 - 1, 2**32 - 2])


def test_two_channel_i16():
    """ Test loading a DAQmx file with two channels of I16 data
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2]))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int16
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_two_channels_without_daqmx_toc_flag():
    """ Test loading a DAQmx file with two channels of DAQmx data but where
        the segment header is missing the DAQmx flag (see issue #226)
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2]))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int16
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_daqmx_metadata_without_daqmx_raw_data():
    """ Test loading a file that uses DAQmx format metadata but where the data types are not raw data (see issue #226)
    """

    scaler_1 = daqmx_scaler_metadata(0, 0xFFFFFFFF, 0)  # Timestamp data
    scaler_2 = daqmx_scaler_metadata(0, 7, 16)
    scaler_3 = daqmx_scaler_metadata(0, 9, 24)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [32], [scaler_1], data_type=types.TimeStamp.enum_value),
        daqmx_channel_metadata("Channel2", 4, [32], [scaler_2], data_type=types.Int64.enum_value),
        daqmx_channel_metadata("Channel3", 4, [32], [scaler_3], data_type=types.DoubleFloat.enum_value))

    times = [
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]

    data = (
        timestamp_hexlify(times[0]) +
        "01 00 00 00 00 00 00 00" +
        hexlify_value("<d", 1.0) +
        timestamp_hexlify(times[1]) +
        "02 00 00 00 00 00 00 00" +
        hexlify_value("<d", 2.0) +
        timestamp_hexlify(times[2]) +
        "03 00 00 00 00 00 00 00" +
        hexlify_value("<d", 3.0) +
        timestamp_hexlify(times[3]) +
        "04 00 00 00 00 00 00 00" +
        hexlify_value("<d", 4.0)
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"][:]
    np.testing.assert_array_equal(data_1, times)

    data_2 = tdms_data["Group"]["Channel2"][:]
    assert data_2.dtype == np.int64
    np.testing.assert_array_equal(data_2, [1, 2, 3, 4])

    data_3 = tdms_data["Group"]["Channel3"][:]
    assert data_3.dtype == np.float64
    np.testing.assert_array_equal(data_3, [1.0, 2.0, 3.0, 4.0])


def test_exception_on_mismatch_of_types_for_non_raw_daqmx():
    scaler = daqmx_scaler_metadata(0, 3, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 2, [2], [scaler], data_type=types.Uint16.enum_value))
    data = "01 00 02 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert error_message == "Expected scaler data type to be Uint16 but got Int16"


def test_exception_on_multiple_scalers_for_non_raw_daqmx():
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 2, [4], [scaler_1, scaler_2], data_type=types.Int32.enum_value))
    data = "01 00 00 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert error_message == "Expected only one scaler for channel with type Int32"


def test_mixed_channel_widths():
    """ Test loading a DAQmx file with channels with different widths
    """

    scaler_1 = daqmx_scaler_metadata(0, 1, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 1)
    scaler_3 = daqmx_scaler_metadata(0, 5, 3)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [7], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [7], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [7], [scaler_3]))
    data = (
        # Data for segment
        "01 11 00 21 00 00 00"
        "02 12 00 22 00 00 00"
        "03 13 00 23 00 00 00"
        "04 14 00 24 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int8
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])

    data_3 = tdms_data["Group"]["Channel3"].raw_data
    assert data_3.dtype == np.int32
    np.testing.assert_array_equal(data_3, [33, 34, 35, 36])


def test_multiple_scalers_with_same_type():
    """ Test loading a DAQmx file with one channel containing multiple
        format changing scalers of the same type
    """

    scaler_metadata = [
        daqmx_scaler_metadata(0, 3, 0),
        daqmx_scaler_metadata(1, 3, 2)]
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], scaler_metadata))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]

    scaler_0_data = channel.raw_scaler_data[0]
    assert scaler_0_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_0_data, [1, 2, 3, 4])

    scaler_1_data = channel.raw_scaler_data[1]
    assert scaler_1_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_1_data, [17, 18, 19, 20])


def test_multiple_scalers_with_different_types():
    """ Test loading a DAQmx file with one channel containing multiple
        format changing scalers of different types
    """

    scaler_metadata = [
        daqmx_scaler_metadata(0, 1, 0),
        daqmx_scaler_metadata(1, 3, 1),
        daqmx_scaler_metadata(2, 5, 3)]
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [7], scaler_metadata))
    data = (
        # Data for segment
        "01 11 00 21 00 00 00"
        "02 12 00 22 00 00 00"
        "03 13 00 23 00 00 00"
        "04 14 00 24 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]

    scaler_0_data = channel.raw_scaler_data[0]
    assert scaler_0_data.dtype == np.int8
    np.testing.assert_array_equal(scaler_0_data, [1, 2, 3, 4])

    scaler_1_data = channel.raw_scaler_data[1]
    assert scaler_1_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_1_data, [17, 18, 19, 20])

    scaler_2_data = channel.raw_scaler_data[2]
    assert scaler_2_data.dtype == np.int32
    np.testing.assert_array_equal(scaler_2_data, [33, 34, 35, 36])


def test_multiple_raw_data_buffers():
    """ Test loading a DAQmx file with multiple raw data buffers
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0, 3, 2, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4, 4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4, 4], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [4, 4], [scaler_3]),
        daqmx_channel_metadata("Channel4", 4, [4, 4], [scaler_4]))
    data = (
        "01 00" "02 00" "03 00" "04 00"
        "05 00" "06 00" "07 00" "08 00"
        "09 00" "0A 00" "0B 00" "0C 00"
        "0D 00" "0E 00" "0F 00" "10 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    data_2 = tdms_data["Group"]["Channel2"].raw_data
    data_3 = tdms_data["Group"]["Channel3"].raw_data
    data_4 = tdms_data["Group"]["Channel4"].raw_data

    for data in [data_1, data_2, data_3, data_4]:
        assert data.dtype == np.int16

    np.testing.assert_array_equal(data_1, [1, 3, 5, 7])
    np.testing.assert_array_equal(data_2, [2, 4, 6, 8])
    np.testing.assert_array_equal(data_3, [9, 11, 13, 15])
    np.testing.assert_array_equal(data_4, [10, 12, 14, 16])


def test_multiple_raw_data_buffers_with_different_widths():
    """ DAQmx with raw data buffers with different widths
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 4, 0)
    scaler_4 = daqmx_scaler_metadata(0, 5, 0, 1)
    scaler_5 = daqmx_scaler_metadata(0, 5, 4, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [6, 8], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [6, 8], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [6, 8], [scaler_3]),
        daqmx_channel_metadata("Channel4", 4, [6, 8], [scaler_4]),
        daqmx_channel_metadata("Channel5", 4, [6, 8], [scaler_5]))
    data = (
        "01 00" "02 00" "03 00"
        "04 00" "05 00" "06 00"
        "07 00" "08 00" "09 00"
        "0A 00" "0B 00" "0C 00"
        "0D 00 00 00" "0E 00 00 00"
        "0F 00 00 00" "10 00 00 00"
        "11 00 00 00" "12 00 00 00"
        "13 00 00 00" "14 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    data_2 = tdms_data["Group"]["Channel2"].raw_data
    data_3 = tdms_data["Group"]["Channel3"].raw_data
    data_4 = tdms_data["Group"]["Channel4"].raw_data
    data_5 = tdms_data["Group"]["Channel5"].raw_data

    for data in [data_1, data_2, data_3]:
        assert data.dtype == np.int16
    for data in [data_4, data_5]:
        assert data.dtype == np.int32

    np.testing.assert_array_equal(data_1, [1, 4, 7, 10])
    np.testing.assert_array_equal(data_2, [2, 5, 8, 11])
    np.testing.assert_array_equal(data_3, [3, 6, 9, 12])
    np.testing.assert_array_equal(data_4, [13, 15, 17, 19])
    np.testing.assert_array_equal(data_5, [14, 16, 18, 20])


def test_multiple_raw_data_buffers_with_different_lengths():
    """ DAQmx with raw data buffers with different lengths
    """

    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 5, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0, 5, 2, 1)
    scaler_5 = daqmx_scaler_metadata(0, 3, 0, 2)
    scaler_6 = daqmx_scaler_metadata(0, 5, 2, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [6, 6, 6], [scaler_1], properties=properties),
        daqmx_channel_metadata("Channel2", 4, [6, 6, 6], [scaler_2], properties=properties),
        daqmx_channel_metadata("Channel3", 2, [6, 6, 6], [scaler_3], properties=properties),
        daqmx_channel_metadata("Channel4", 2, [6, 6, 6], [scaler_4], properties=properties),
        daqmx_channel_metadata("Channel5", 1, [6, 6, 6], [scaler_5], properties=properties),
        daqmx_channel_metadata("Channel6", 1, [6, 6, 6], [scaler_6], properties=properties))
    data = (
        # Chunk 1
        # Buffer 0
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        # Buffer 1
        "03 00" "04 00 00 00"
        "03 00" "04 00 00 00"
        # Buffer 2
        "05 00" "06 00 00 00"
        # Chunk 2
        # Buffer 0
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        # Buffer 1
        "03 00" "04 00 00 00"
        "03 00" "04 00 00 00"
        # Buffer 2
        "05 00" "06 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"][:]
    data_2 = tdms_data["Group"]["Channel2"][:]
    data_3 = tdms_data["Group"]["Channel3"][:]
    data_4 = tdms_data["Group"]["Channel4"][:]
    data_5 = tdms_data["Group"]["Channel5"][:]
    data_6 = tdms_data["Group"]["Channel6"][:]

    for data in [data_1, data_3, data_5]:
        assert data.dtype == np.int16
    for data in [data_2, data_4, data_6]:
        assert data.dtype == np.int32

    np.testing.assert_array_equal(data_1, [1] * 8)
    np.testing.assert_array_equal(data_2, [2] * 8)
    np.testing.assert_array_equal(data_3, [3] * 4)
    np.testing.assert_array_equal(data_4, [4] * 4)
    np.testing.assert_array_equal(data_5, [5] * 2)
    np.testing.assert_array_equal(data_6, [6] * 2)


def test_incomplete_segment_with_different_length_buffers():
    """ DAQmx with raw data buffers with different lengths
    """

    scaler_1 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 1)
    scaler_5 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 2)
    scaler_6 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4, 4, 4], [scaler_1], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel2", 4, [4, 4, 4], [scaler_2], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel3", 2, [4, 4, 4], [scaler_3], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel4", 2, [4, 4, 4], [scaler_4], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel5", 1, [4, 4, 4], [scaler_5], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel6", 1, [4, 4, 4], [scaler_6], data_type=types.Uint16.enum_value))
    data = (
        # Chunk 1
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
        "03 00" "04 00"
        # Buffer 2
        "05 00" "06 00"
        # Chunk 2
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
        "03 00" "04 00"
        # Buffer 2
        "05 00" "06 00"
        # Incomplete third chunk
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data, incomplete=True)
    tdms_data = test_file.load()

    group = tdms_data["Group"]

    np.testing.assert_array_equal(group["Channel1"][:], [1] * 12)
    np.testing.assert_array_equal(group["Channel2"][:], [2] * 12)
    np.testing.assert_array_equal(group["Channel3"][:], [3] * 5)
    np.testing.assert_array_equal(group["Channel4"][:], [4] * 5)
    np.testing.assert_array_equal(group["Channel5"][:], [5] * 2)
    np.testing.assert_array_equal(group["Channel6"][:], [6] * 2)

    file_status = tdms_data.file_status
    assert file_status.incomplete_final_segment
    assert file_status.channel_statuses is not None
    for channel in ["Channel1", "Channel2"]:
        channel_status = file_status.channel_statuses[f"/'Group'/'{channel}'"]
        assert channel_status.expected_length == 4
        assert channel_status.read_length == 4
    for channel in ["Channel3", "Channel4"]:
        channel_status = file_status.channel_statuses[f"/'Group'/'{channel}'"]
        assert channel_status.expected_length == 2
        assert channel_status.read_length == 1
    for channel in ["Channel5", "Channel6"]:
        channel_status = file_status.channel_statuses[f"/'Group'/'{channel}'"]
        assert channel_status.expected_length == 1
        assert channel_status.read_length == 0


def test_multiple_raw_data_buffers_with_scalers_split_across_buffers():
    """ DAQmx with scalers split across different raw data buffers
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(1, 3, 0, 1)
    scaler_3 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_4 = daqmx_scaler_metadata(1, 3, 2, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata(
            "Channel1", 4, [4, 4], [scaler_1, scaler_2]),
        daqmx_channel_metadata(
            "Channel2", 4, [4, 4], [scaler_3, scaler_4]))
    data = (
        "01 00" "02 00" "03 00" "04 00"
        "05 00" "06 00" "07 00" "08 00"
        "09 00" "0A 00" "0B 00" "0C 00"
        "0D 00" "0E 00" "0F 00" "10 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    channel_1 = tdms_data["Group"]["Channel1"]
    channel_2 = tdms_data["Group"]["Channel2"]

    scaler_data_1 = channel_1.raw_scaler_data[0]
    scaler_data_2 = channel_1.raw_scaler_data[1]
    scaler_data_3 = channel_2.raw_scaler_data[0]
    scaler_data_4 = channel_2.raw_scaler_data[1]

    for data in [
            scaler_data_1, scaler_data_2, scaler_data_3, scaler_data_4]:
        assert data.dtype == np.int16

    np.testing.assert_array_equal(scaler_data_1, [1, 3, 5, 7])
    np.testing.assert_array_equal(scaler_data_2, [9, 11, 13, 15])
    np.testing.assert_array_equal(scaler_data_3, [2, 4, 6, 8])
    np.testing.assert_array_equal(scaler_data_4, [10, 12, 14, 16])


def test_digital_line_scaler_data():
    """ Test loading a DAQmx file with a single channel of U8 digital line scaler data
    """

    scaler_metadata = digital_scaler_metadata(0, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata], digital_line_scaler=True))
    data = (
        "00 00 00 00"
        "01 00 00 00"
        "00 00 00 00"
        "01 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, [0, 1, 0, 1])


@pytest.mark.parametrize('byte_offset', [0, 1, 2, 3])
def test_digital_line_scaler_with_multiple_channels(byte_offset):
    """ Test DAQmx digital line scaler data with multiple channels
    """

    scaler_metadata_0 = digital_scaler_metadata(0, 0, byte_offset * 8 + 0)
    scaler_metadata_1 = digital_scaler_metadata(0, 0, byte_offset * 8 + 1)
    scaler_metadata_2 = digital_scaler_metadata(0, 0, byte_offset * 8 + 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel0", 4, [4], [scaler_metadata_0], digital_line_scaler=True),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata_1], digital_line_scaler=True),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_metadata_2], digital_line_scaler=True),
    )
    byte_values = [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
    ]
    hex_data = " ".join("00" * byte_offset + b + "00" * (3 - byte_offset) for b in byte_values)

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, hex_data)
    tdms_data = test_file.load()

    for (channel_name, expected_data) in [
        ("Channel0", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("Channel1", [0, 0, 1, 1, 0, 0, 1, 1]),
        ("Channel2", [0, 0, 0, 0, 1, 1, 1, 1]),
    ]:
        data = tdms_data["Group"][channel_name].raw_data

        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, expected_data, "Incorrect data for channel '%s'" % channel_name)


def test_exception_on_mismatch_of_raw_data_widths():
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 1, [2], [scaler_1]),
        daqmx_channel_metadata("Channel2", 1, [4], [scaler_2]))
    data = "01 00 02 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert (error_message == "Raw data widths for object DaqmxSegmentObject(/'Group'/'Channel2') ([4]) "
            "do not match previous widths ([2])")


def test_lazily_reading_channel():
    """ Test loading channels individually from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data()
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

            data_2 = tdms_file["Group"]["Channel2"].read_data()
            assert data_2.dtype == np.int16
            np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_lazily_reading_a_subset_of_channel_data():
    """ Test loading a subset of channel data from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data(1, 2)
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [2, 3])

            data_2 = tdms_file["Group"]["Channel2"].read_data(1, 2)
            assert data_2.dtype == np.int16
            np.testing.assert_array_equal(data_2, [18, 19])


def test_lazily_reading_a_subset_of_raw_channel_data():
    """ Test loading a subset of raw scaler channel data from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data(1, 2, scaled=False)
            assert len(data_1) == 1
            assert data_1[0].dtype == np.int16
            np.testing.assert_array_equal(data_1[0], [2, 3])

            data_2 = tdms_file["Group"]["Channel2"].read_data(1, 2, scaled=False)
            assert len(data_2) == 1
            assert data_2[0].dtype == np.int16
            np.testing.assert_array_equal(data_2[0], [18, 19])


@pytest.mark.parametrize('offset,length', [
    (0, None),
    (1, None),
    (0, 2),
    (1, 2),
])
def test_read_raw_data(offset, length):
    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(1, 3, 0)
    scaler_2 = daqmx_scaler_metadata(2, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    end = None if length is None else offset + length
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        data_1 = tdms_file["Group"]["Channel1"].read_data(offset=offset, length=length, scaled=False)
        assert data_1[1].dtype == np.int16
        np.testing.assert_array_equal(data_1[1], [1, 2, 3, 4][offset:end])

        data_2 = tdms_file["Group"]["Channel2"].read_data(offset=offset, length=length, scaled=False)
        assert data_2[2].dtype == np.int16
        np.testing.assert_array_equal(data_2[2], [17, 18, 19, 20][offset:end])


def test_stream_data_chunks():
    """Test streaming chunks of DAQmx data from a TDMS file
    """
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00" "11 00"
        "02 00" "12 00"
        "03 00" "13 00"
        "04 00" "14 00"
        "05 00" "15 00"
        "06 00" "16 00"
        "07 00" "17 00"
        "08 00" "18 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    data_arrays = defaultdict(list)
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for chunk in tdms_file.data_chunks():
                for group in chunk.groups():
                    for channel in group.channels():
                        key = (group.name, channel.name)
                        assert channel.offset == len(data_arrays[key])
                        data_arrays[key].extend(channel[:])

    expected_channel_data = {
        ("Group", "Channel1"): [1, 2, 3, 4, 5, 6, 7, 8],
        ("Group", "Channel2"): [17, 18, 19, 20, 21, 22, 23, 24],
    }
    for ((group, channel), expected_data) in expected_channel_data.items():
        actual_data = data_arrays[(group, channel)]
        np.testing.assert_equal(actual_data, expected_data)


def test_daqmx_debug_logging(caplog):
    """ Test loading a DAQmx file with debug logging enabled
    """
    scaler_metadata = daqmx_scaler_metadata(0, 3, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    log_manager.set_level(logging.DEBUG)
    _ = test_file.load()

    assert "Reading metadata for object /'Group'/'Channel1' with index header 0x00001269" in caplog.text
    assert "scale_id=0" in caplog.text
    assert "data_type=Int16" in caplog.text


def segment_toc():
    return (
        "kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocDAQmxRawData")


def segment_toc_non_daqmx():
    return (
        "kTocMetaData", "kTocRawData", "kTocNewObjList")


def daqmx_scaler_metadata(scale_id, type_id, byte_offset, raw_buffer_index=0):
    return (
        # DAQmx data type (type ids don't match TDMS types)
        hexlify_value("<I", type_id) +
        # Raw buffer index
        hexlify_value("<I", raw_buffer_index) +
        # Raw byte offset
        hexlify_value("<I", byte_offset) +
        # Sample format bitmap (don't know what this is for...)
        "00 00 00 00" +
        # Scale ID
        hexlify_value("<I", scale_id))


def digital_scaler_metadata(scale_id, type_id, bit_offset, raw_buffer_index=0):
    return (
        # DAQmx data type (type ids don't match TDMS types)
        hexlify_value("<I", type_id) +
        # Raw buffer index
        hexlify_value("<I", raw_buffer_index) +
        # Raw byte offset
        hexlify_value("<I", bit_offset) +
        # Sample format bitmap (don't know what this is for...)
        "00" +
        # Scale ID
        hexlify_value("<I", scale_id))


def daqmx_channel_metadata(
        channel_name, num_values,
        raw_data_widths, scaler_metadata, properties=None, digital_line_scaler=False,
        data_type=None):
    path = "/'Group'/'" + channel_name + "'"
    # Default to DAQmx data type
    data_type = 0xFFFFFFFF if data_type is None else data_type
    return (
        # Length of the object path
        hexlify_value("<I", len(path)) +
        # Object path
        string_hexlify(path) +
        # Raw data index (DAQmx)
        ("6A 12 00 00" if digital_line_scaler else "69 12 00 00") +
        # Data type
        hexlify_value("<I", data_type) +
        # Array  dimension
        "01 00 00 00" +
        # Number of values (chunk size)
        hexlify_value("<Q", num_values) +
        # Scaler metadata
        hexlify_value("<I", len(scaler_metadata)) +
        "".join(scaler_metadata) +
        # Raw data width vector size
        hexlify_value("<I", len(raw_data_widths)) +
        # Raw data width values
        "".join(hexlify_value("<I", v) for v in raw_data_widths) +
        hex_properties(properties))


def daqmx_channel_scaling_properties(number_of_scales, scale_properties=None):
    """Build channel properties that configure DAQmx scaling

    :param number_of_scales: Value of the NI_Number_Of_Scales property
    :param scale_properties: Dictionary of additional property name to
        (property type, hex encoded value) pairs.
    """
    properties = {
        "NI_Number_Of_Scales": (3, hexlify_value("<i", number_of_scales)),
    }
    if scale_properties is not None:
        properties.update(scale_properties)
    return properties


def daqmx_string_property(value):
    """Hex encoded value for a string TDMS property"""
    encoded = value.encode("utf-8")
    return hexlify_value("<I", len(encoded)) + string_hexlify(value)


def daqmx_int_property(value):
    """Hex encoded value for an int32 TDMS property"""
    return hexlify_value("<i", value)


def daqmx_add_scaling_properties(left_source, right_source, scale_index=2):
    """Channel properties for an Add scale reading two other scales"""
    prefix = "NI_Scale[%d]" % scale_index
    return daqmx_channel_scaling_properties(scale_index + 1, {
        prefix + "_Scale_Type": (0x20, daqmx_string_property("Add")),
        prefix + "_Add_Left_Operand_Input_Source": (3, daqmx_int_property(left_source)),
        prefix + "_Add_Right_Operand_Input_Source": (3, daqmx_int_property(right_source)),
    })


def _split_buffer_data(buffers):
    """Join per-buffer lists of values into the byte string for one chunk

    :param buffers: List of per-buffer byte strings, in buffer order.
    """
    return "".join(buffers)


def _i16_data(values):
    return "".join(hexlify_value("<h", v) for v in values)


def _truncated_two_buffer_file(
        channel_c_scaler_1_values, channel_a_values, extra_segments=0):
    """Build a DAQmx file with complete chunks then an incomplete final chunk

    Channel C has scaler 0 from buffer 0 and scaler 1 from buffer 1, and its
    final scale adds the two scalers. Channel A only has a scaler in buffer 0.

    :param channel_c_scaler_1_values: Values in buffer 1 of the incomplete chunk.
    :param channel_a_values: Values in buffer 0 of the incomplete chunk.
    :param extra_segments: Number of additional complete chunks to write before
        the incomplete final chunk.
    """
    c_properties = daqmx_add_scaling_properties(0, 1)
    a_properties = daqmx_channel_scaling_properties(1)

    c_scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    c_scaler_1 = daqmx_scaler_metadata(1, 3, 0, 1)
    a_scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("C", 4, [2, 2], [c_scaler_0, c_scaler_1], properties=c_properties),
        daqmx_channel_metadata("A", 4, [2, 2], [a_scaler_0], properties=a_properties))

    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40])
    final_data = _i16_data(channel_a_values) + _i16_data(channel_c_scaler_1_values)

    test_file = GeneratedFile()
    for _ in range(extra_segments + 1):
        test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, final_data, incomplete=True)
    return test_file


def test_truncated_final_chunk_with_scalers_in_different_buffers():
    """ A truncated DAQmx chunk can compute a sample only when all inputs used by
    the final scaling have a value.

    Channel C adds scaler 0 (buffer 0) and scaler 1 (buffer 1). Buffer 0 is
    complete for the final chunk, buffer 1 only has one value, so only the
    first sample of the final chunk is computable. The extra buffer 0 values
    must not be broadcast against the single buffer 1 value.
    """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8])

    tdms_data = test_file.load()
    channel_c = tdms_data["Group"]["C"]
    channel_a = tdms_data["Group"]["A"]

    np.testing.assert_array_equal(channel_c[:], [11, 22, 33, 44, 55])
    np.testing.assert_array_equal(channel_c.data, [11, 22, 33, 44, 55])
    # Only the common computable range is delivered for the scalers used by the
    # final scaling; the extra raw buffer 0 values are not padded or broadcast.
    np.testing.assert_array_equal(channel_c.raw_scaler_data[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(channel_c.raw_scaler_data[1], [10, 20, 30, 40, 50])

    # Channel A doesn't depend on buffer 1, so none of its samples are lost
    np.testing.assert_array_equal(channel_a[:], [1, 2, 3, 4, 5, 6, 7, 8])

    file_status = tdms_data.file_status
    assert file_status.incomplete_final_segment
    c_status = file_status.channel_statuses["/'Group'/'C'"]
    assert c_status.expected_length == 4
    assert c_status.read_length == 1
    a_status = file_status.channel_statuses["/'Group'/'A'"]
    assert a_status.expected_length == 4
    assert a_status.read_length == 4

    # The undelivered difference is reported: buffer 0 had four complete raw
    # values but only one could be paired with buffer 1.
    assert c_status.scaler_statuses[0].available_length == 4
    assert c_status.scaler_statuses[0].read_length == 1
    assert c_status.scaler_statuses[1].available_length == 1
    assert c_status.scaler_statuses[1].read_length == 1


def test_truncated_final_chunk_lengths_match_across_entry_points():
    """ read, read_metadata and lazy channels must agree on the computable range """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8])

    read_file = test_file.load()
    metadata_file = TdmsFile.read_metadata(test_file.get_bytes_io_file())

    assert len(read_file["Group"]["C"]) == 5
    assert len(metadata_file["Group"]["C"]) == 5
    assert len(read_file["Group"]["A"]) == 8
    assert len(metadata_file["Group"]["A"]) == 8

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        assert len(lazy_file["Group"]["C"]) == 5
        assert len(lazy_file["Group"]["A"]) == 8
        np.testing.assert_array_equal(lazy_file["Group"]["C"][:], [11, 22, 33, 44, 55])
        np.testing.assert_array_equal(lazy_file["Group"]["A"][:], [1, 2, 3, 4, 5, 6, 7, 8])

        for channel_name in ["C", "A"]:
            channel = lazy_file["Group"][channel_name]
            data = channel.read_data()
            np.testing.assert_array_equal(data, read_file["Group"][channel_name][:])

        # Indexing and slicing the last values and across the chunk boundary
        assert lazy_file["Group"]["C"][-1] == 55
        assert lazy_file["Group"]["C"][-2] == 44
        np.testing.assert_array_equal(lazy_file["Group"]["C"][-2:], [44, 55])
        np.testing.assert_array_equal(lazy_file["Group"]["C"][3:5], [44, 55])
        np.testing.assert_array_equal(lazy_file["Group"]["A"][3:7], [4, 5, 6, 7])


def test_truncated_final_chunk_file_data_chunks():
    """ File level chunk offsets and lengths must follow the recovered range """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8])

    offsets = {"C": [], "A": []}
    chunks = {"C": [], "A": []}
    with TdmsFile.open(test_file.get_bytes_io_file()) as tdms_file:
        for raw_chunk in tdms_file.data_chunks():
            for channel_name in ["C", "A"]:
                channel_chunk = raw_chunk["Group"][channel_name]
                offsets[channel_name].append(channel_chunk.offset)
                chunks[channel_name].append(channel_chunk[:])

    assert offsets["C"] == [0, 4]
    np.testing.assert_array_equal(chunks["C"][0], [11, 22, 33, 44])
    np.testing.assert_array_equal(chunks["C"][1], [55])
    assert offsets["A"] == [0, 4]
    np.testing.assert_array_equal(chunks["A"][0], [1, 2, 3, 4])
    np.testing.assert_array_equal(chunks["A"][1], [5, 6, 7, 8])


def test_truncated_final_chunk_channel_data_chunks():
    """ Channel level chunk offsets and lengths must follow the recovered range """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8])

    with TdmsFile.open(test_file.get_bytes_io_file()) as tdms_file:
        for channel_name, expected_chunks, expected_offsets in [
                ("C", [[11, 22, 33, 44], [55]], [0, 4]),
                ("A", [[1, 2, 3, 4], [5, 6, 7, 8]], [0, 4])]:
            channel = tdms_file["Group"][channel_name]
            actual_chunks = []
            actual_offsets = []
            for chunk in channel.data_chunks():
                actual_chunks.append(chunk[:])
                actual_offsets.append(chunk.offset)
            assert actual_offsets == expected_offsets
            assert len(actual_chunks) == len(expected_chunks)
            for (actual, expected) in zip(actual_chunks, expected_chunks):
                np.testing.assert_array_equal(actual, expected)


def test_truncated_final_chunk_raw_scaler_lengths_across_read_modes():
    """ Raw scaler lengths are consistent for read, lazy and memmap reading """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8])

    def check_channel(channel):
        np.testing.assert_array_equal(channel.raw_scaler_data[0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(channel.raw_scaler_data[1], [10, 20, 30, 40, 50])
        raw_data = channel.read_data(scaled=False)
        np.testing.assert_array_equal(raw_data[0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(raw_data[1], [10, 20, 30, 40, 50])

    check_channel(test_file.load()["Group"]["C"])

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            np.testing.assert_array_equal(tdms_file["Group"]["C"].read_data(scaled=False)[0],
                                          [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(tdms_file["Group"]["C"].read_data(scaled=False)[1],
                                          [10, 20, 30, 40, 50])

    with tempfile.TemporaryDirectory() as memmap_dir:
        check_channel(test_file.load(memmap_dir=memmap_dir)["Group"]["C"])


def test_truncated_final_chunk_ending_on_buffer_boundary():
    """ Final chunk ends exactly on a buffer boundary: buffer 0 complete, buffer 1 absent """
    test_file = _truncated_two_buffer_file([], [5, 6, 7, 8])

    tdms_data = test_file.load()
    np.testing.assert_array_equal(tdms_data["Group"]["C"][:], [11, 22, 33, 44])
    np.testing.assert_array_equal(tdms_data["Group"]["A"][:], [1, 2, 3, 4, 5, 6, 7, 8])
    # The unused raw buffer 0 tail is reported, not silently dropped
    np.testing.assert_array_equal(tdms_data["Group"]["C"].raw_scaler_data[0], [1, 2, 3, 4])

    c_status = tdms_data.file_status.channel_statuses["/'Group'/'C'"]
    assert c_status.read_length == 0
    assert c_status.scaler_statuses[0].available_length == 4
    assert c_status.scaler_statuses[0].read_length == 0
    assert c_status.scaler_statuses[1].available_length == 0
    assert c_status.scaler_statuses[1].read_length == 0

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        assert len(lazy_file["Group"]["C"]) == 4
        assert len(lazy_file["Group"]["A"]) == 8


@pytest.mark.parametrize("partial_buffer_1_bytes, expected_c_samples", [
    (["32"], []),                          # Half of a value, no complete value
    (["32", "00", "33"], [55]),            # One complete value and a half, the tail byte is unusable
])
def test_truncated_final_chunk_ending_mid_value(partial_buffer_1_bytes, expected_c_samples):
    """ A partial trailing value must not be used to compute a sample """
    c_properties = daqmx_add_scaling_properties(0, 1)
    a_properties = daqmx_channel_scaling_properties(1)
    c_scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    c_scaler_1 = daqmx_scaler_metadata(1, 3, 0, 1)
    a_scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("C", 4, [2, 2], [c_scaler_0, c_scaler_1], properties=c_properties),
        daqmx_channel_metadata("A", 4, [2, 2], [a_scaler_0], properties=a_properties))
    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40])
    partial_data = _i16_data([5, 6, 7, 8]) + "".join(partial_buffer_1_bytes)

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, partial_data, incomplete=True)

    expected_c = [11, 22, 33, 44] + expected_c_samples
    tdms_data = test_file.load()
    np.testing.assert_array_equal(tdms_data["Group"]["C"][:], expected_c)
    np.testing.assert_array_equal(tdms_data["Group"]["A"][:], [1, 2, 3, 4, 5, 6, 7, 8])

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        assert len(lazy_file["Group"]["C"]) == len(expected_c)
        np.testing.assert_array_equal(lazy_file["Group"]["C"][:], expected_c)
        # The raw scaler holds only the values in the computable range and never
        # includes the incomplete trailing bytes
        raw_scaler_1 = lazy_file["Group"]["C"].read_data(scaled=False)[1]
        assert len(raw_scaler_1) == len(expected_c)
        np.testing.assert_array_equal(raw_scaler_1, [10, 20, 30, 40] + [50] * len(expected_c_samples))
        assert lazy_file.file_status.channel_statuses["/'Group'/'C'"].scaler_statuses[1].available_length \
            == len(expected_c_samples)


def test_truncated_final_chunk_after_multiple_complete_chunks():
    """ Slices, last index and chunk offsets are correct after several complete chunks """
    test_file = _truncated_two_buffer_file([50], [5, 6, 7, 8], extra_segments=2)

    with TdmsFile.open(test_file.get_bytes_io_file()) as tdms_file:
        channel_c = tdms_file["Group"]["C"]
        assert len(channel_c) == 13
        np.testing.assert_array_equal(
            channel_c[:], [11, 22, 33, 44] * 3 + [55])
        np.testing.assert_array_equal(channel_c[3:9], [44, 11, 22, 33, 44, 11])
        np.testing.assert_array_equal(channel_c[-5:], [11, 22, 33, 44, 55])
        assert channel_c[-1] == 55
        assert channel_c[12] == 55
        assert channel_c[13 - 1] == 55

        channel_a = tdms_file["Group"]["A"]
        assert len(channel_a) == 16
        np.testing.assert_array_equal(
            channel_a[:], [1, 2, 3, 4] * 3 + [5, 6, 7, 8])

        chunks = list(channel_c.data_chunks())
        assert [chunk.offset for chunk in chunks] == [0, 4, 8, 12]
        assert [len(chunk) for chunk in chunks] == [4, 4, 4, 1]
        for chunk in chunks:
            np.testing.assert_array_equal(
                chunk[:], channel_c[chunk.offset:chunk.offset + len(chunk)])

        file_offsets = []
        for raw_chunk in tdms_file.data_chunks():
            file_offsets.append(raw_chunk["Group"]["C"].offset)
        assert file_offsets == [0, 4, 8, 12]


def test_truncated_final_chunk_with_unused_short_scaler():
    """ A scaler not read by the final scaling must not shorten the channel """
    properties = daqmx_channel_scaling_properties(1)
    # Scaler 0 is read by the final scale; scaler 1 exists but is unused.
    # They live in different buffers so the unused one can be short.
    used_scaler = daqmx_scaler_metadata(0, 3, 0, 0)
    unused_scaler = daqmx_scaler_metadata(1, 3, 0, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2, 2], [used_scaler, unused_scaler], properties=properties))
    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40])
    final_data = _i16_data([5, 6, 7, 8]) + _i16_data([50])

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, final_data, incomplete=True)

    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]
    np.testing.assert_array_equal(channel[:], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(channel.raw_scaler_data[0], [1, 2, 3, 4, 5, 6, 7, 8])
    # The unused scaler keeps its own, shorter length rather than being padded
    np.testing.assert_array_equal(channel.raw_scaler_data[1], [10, 20, 30, 40, 50])

    c_status = tdms_data.file_status.channel_statuses["/'Group'/'Channel1'"]
    assert c_status.expected_length == 4
    assert c_status.read_length == 4
    assert c_status.scaler_statuses[1].available_length == 1
    assert c_status.scaler_statuses[1].read_length == 1

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        lazy_channel = lazy_file["Group"]["Channel1"]
        assert len(lazy_channel) == 8
        np.testing.assert_array_equal(lazy_channel[:], [1, 2, 3, 4, 5, 6, 7, 8])
        raw_data = lazy_channel.read_data(scaled=False)
        assert len(raw_data[0]) == 8
        np.testing.assert_array_equal(raw_data[1], [10, 20, 30, 40, 50])


def test_truncated_final_chunk_with_final_scale_using_one_input():
    """ A final scale that only reads one scaler doesn't depend on the other input """
    properties = daqmx_add_scaling_properties(0, 0)
    scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_1 = daqmx_scaler_metadata(1, 3, 0, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2, 2], [scaler_0, scaler_1], properties=properties))
    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40])
    final_data = _i16_data([5, 6, 7, 8]) + _i16_data([50])

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, final_data, incomplete=True)

    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]
    # Final scale is scale 0 added to itself, so no buffer 1 values are needed
    np.testing.assert_array_equal(channel[:], [2, 4, 6, 8, 10, 12, 14, 16])
    # Buffer 1 values are kept only where they were complete
    np.testing.assert_array_equal(channel.raw_scaler_data[1], [10, 20, 30, 40, 50])
    status = tdms_data.file_status.channel_statuses["/'Group'/'Channel1'"]
    assert status.read_length == 4
    assert status.scaler_statuses[1].available_length == 1
    assert status.scaler_statuses[1].read_length == 1


def test_truncated_final_chunk_with_different_buffer_widths_and_types():
    """ Buffers with different widths and integer types are handled independently """
    properties = daqmx_channel_scaling_properties(1)
    # Buffer 0 is an int16 scaler and buffer 1 is an int32 scaler with a wider row.
    # The final scale reads scale 0; scale 5 exists but is not used by it.
    scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_1 = daqmx_scaler_metadata(5, 5, 0, 1)
    # A second channel only reads buffer 0 and must keep all of its samples
    independent_scaler = daqmx_scaler_metadata(0, 3, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Wide", 4, [2, 4], [scaler_0, scaler_1], properties=properties),
        daqmx_channel_metadata("BufferZero", 4, [2, 4], [independent_scaler], properties=properties))
    complete_data = _i16_data([1, 2, 3, 4]) + "".join(hexlify_value("<i", v) for v in [10, 20, 30, 40])
    # Final chunk has all of buffer 0 and one complete int32 in buffer 1
    final_data = _i16_data([5, 6, 7, 8]) + hexlify_value("<i", 50)

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, final_data, incomplete=True)

    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Wide"]
    # The final scaling only reads scaler 0, so all eight buffer 0 values are usable
    np.testing.assert_array_equal(channel[:], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(channel.raw_scaler_data[0], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(channel.raw_scaler_data[5], [10, 20, 30, 40, 50])
    assert channel.raw_scaler_data[5].dtype == np.int32

    np.testing.assert_array_equal(
        tdms_data["Group"]["BufferZero"][:], [1, 2, 3, 4, 5, 6, 7, 8])

    status = tdms_data.file_status.channel_statuses["/'Group'/'Wide'"]
    assert status.read_length == 4
    assert status.scaler_statuses[5].available_length == 1


def test_truncated_final_chunk_without_metadata_block():
    """ A truncated final segment reusing the previous structure still gets the scaling right """
    c_properties = daqmx_add_scaling_properties(0, 1)
    a_properties = daqmx_channel_scaling_properties(1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata(
            "C", 4, [2, 2],
            [daqmx_scaler_metadata(0, 3, 0, 0), daqmx_scaler_metadata(1, 3, 0, 1)],
            properties=c_properties),
        daqmx_channel_metadata(
            "A", 4, [2, 2], [daqmx_scaler_metadata(0, 3, 0, 0)], properties=a_properties))
    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40])
    final_data = _i16_data([5, 6, 7, 8]) + _i16_data([50])

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    # Final segment has no metadata block and reuses the previous object structure
    test_file.add_segment(("kTocRawData",), "", final_data, incomplete=True)

    tdms_data = test_file.load()
    np.testing.assert_array_equal(tdms_data["Group"]["C"][:], [11, 22, 33, 44, 55])
    np.testing.assert_array_equal(tdms_data["Group"]["A"][:], [1, 2, 3, 4, 5, 6, 7, 8])

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        assert len(lazy_file["Group"]["C"]) == 5
        np.testing.assert_array_equal(lazy_file["Group"]["C"][:], [11, 22, 33, 44, 55])
        status = lazy_file.file_status.channel_statuses["/'Group'/'C'"]
        assert status.read_length == 1


def test_truncated_final_chunk_with_three_buffers():
    """ A channel scaled from a later buffer keeps only that buffer's computable range """
    properties = daqmx_channel_scaling_properties(1)
    scaler_0 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 1)
    scaler_2 = daqmx_scaler_metadata(0, 3, 0, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata(
            "Ch", 4, [2, 2, 2], [scaler_0, scaler_1, scaler_2], properties=properties))
    complete_data = _i16_data([1, 2, 3, 4]) + _i16_data([10, 20, 30, 40]) + _i16_data([100, 200, 300, 400])
    final_data = _i16_data([5, 6, 7, 8]) + _i16_data([50, 60, 70, 80]) + _i16_data([500, 600])

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, complete_data)
    test_file.add_segment(segment_toc(), metadata, final_data, incomplete=True)

    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Ch"]
    np.testing.assert_array_equal(channel[:], [100, 200, 300, 400, 500, 600])

    with TdmsFile.open(test_file.get_bytes_io_file()) as lazy_file:
        lazy_channel = lazy_file["Group"]["Ch"]
        assert len(lazy_channel) == 6
        np.testing.assert_array_equal(lazy_channel[-3:], [400, 500, 600])
        assert [chunk.offset for chunk in lazy_channel.data_chunks()] == [0, 4]
        status = lazy_file.file_status.channel_statuses["/'Group'/'Ch'"]
        assert status.read_length == 2
