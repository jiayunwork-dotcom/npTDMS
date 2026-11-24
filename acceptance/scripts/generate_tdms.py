"""Generate acceptance-test TDMS files.

Creates three-channel float64 files in both interleaved and contiguous
(non-interleaved) layouts with configurable storage chunk sizes.

Each generated file is a single TDMS segment whose metadata declares one
storage chunk (one "unit" of rows). Repeated storage units are written
without any new metadata, so the segment contains num_units storage chunks
of rows_per_unit rows each.
"""

import argparse
import os
import struct

import numpy as np


TDS_FLOAT64 = 10
TOC_METADATA = 1 << 1
TOC_RAW_DATA = 1 << 3
TOC_NEW_OBJ_LIST = 1 << 2
TOC_INTERLEAVED = 1 << 5


def _encode_string(value):
    raw = value.encode('utf-8')
    return struct.pack('<I', len(raw)) + raw


def _build_metadata(channels, rows_per_unit):
    parts = [struct.pack('<I', len(channels))]
    for path in channels:
        parts.append(_encode_string(path))
        # Length of the raw data index section (20 bytes for fixed size types)
        parts.append(struct.pack('<I', 20))
        # Data type, dimension and number of values per storage chunk
        parts.append(struct.pack('<IIQ', TDS_FLOAT64, 1, rows_per_unit))
        # Number of properties
        parts.append(struct.pack('<I', 0))
    return b''.join(parts)


def _build_lead_in(toc_mask, metadata_size, data_size):
    return (
        b'TDSm'
        + struct.pack('<i', toc_mask)
        + struct.pack('<i', 4713)
        + struct.pack('<Q', metadata_size + data_size)
        + struct.pack('<Q', metadata_size))


def write_file(file_path, channels, data, rows_per_unit, num_units, interleaved,
               truncated_final_rows=None):
    """Write a TDMS file with one segment made up of repeated storage chunks.

    :param data: dict mapping channel path to a 1-D float64 array.
    :param truncated_final_rows: if not None, the final storage chunk is written
        with only this many rows (an incomplete final chunk).
    """
    num_rows = rows_per_unit * num_units
    for values in data.values():
        assert len(values) == num_rows or (
            truncated_final_rows is not None
            and len(values) == rows_per_unit * (num_units - 1) + truncated_final_rows)

    metadata = _build_metadata(channels, rows_per_unit)
    toc_mask = TOC_METADATA | TOC_RAW_DATA | TOC_NEW_OBJ_LIST
    if interleaved:
        toc_mask |= TOC_INTERLEAVED

    data_blocks = []
    for unit in range(num_units):
        is_final = unit == num_units - 1 and truncated_final_rows is not None
        rows_in_unit = truncated_final_rows if is_final else rows_per_unit
        start = unit * rows_per_unit
        end = start + rows_in_unit
        if interleaved:
            # Rows are interleaved: for each row, one value from each channel.
            block = np.empty(rows_in_unit * len(channels), dtype='<f8')
            for row in range(rows_in_unit):
                for col, channel_path in enumerate(channels):
                    block[row * len(channels) + col] = data[channel_path][start + row]
            data_blocks.append(block.tobytes())
        else:
            # Contiguous: a channel's whole block for this unit, then the next channel.
            for channel_path in channels:
                data_blocks.append(data[channel_path][start:end].astype('<f8').tobytes())
    raw_data = b''.join(data_blocks)

    lead_in = _build_lead_in(toc_mask, len(metadata), len(raw_data))
    with open(file_path, 'wb') as f:
        f.write(lead_in)
        f.write(metadata)
        f.write(raw_data)


def make_data(num_rows, num_channels=3, seed=1234):
    rng = np.random.default_rng(seed)
    channels = ["/'group'/'channel%d'" % i for i in range(num_channels)]
    data = {
        path: rng.uniform(-1e6, 1e6, size=num_rows).astype(np.float64)
        for path in channels}
    return channels, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    args = parser.parse_args()
    out_dir = args.out_dir

    cases = [
        # (name, rows_per_unit, num_units)
        ("large", 200_000, 4),       # 800,000 rows, 4 storage units
        ("small", 2, 100_000),       # 200,000 rows, 100,000 storage units
    ]

    for name, rows_per_unit, num_units in cases:
        num_rows = rows_per_unit * num_units
        channels, data = make_data(num_rows)
        for interleaved in (False, True):
            layout = "interleaved" if interleaved else "contiguous"
            path = os.path.join(out_dir, f"{name}_{layout}.tdms")
            write_file(path, channels, data, rows_per_unit, num_units, interleaved)
            print("wrote", path)

    # Files with an incomplete final storage chunk (large-chunk interleaved case)
    truncated_rows = 123
    base_rows = 200_000
    num_units = 4
    total_rows = base_rows * (num_units - 1) + truncated_rows
    channels, data = make_data(total_rows)
    for interleaved in (False, True):
        layout = "interleaved" if interleaved else "contiguous"
        path = os.path.join(out_dir, f"truncated_{layout}.tdms")
        write_file(path, channels, data, base_rows, num_units, interleaved,
                   truncated_final_rows=truncated_rows)
        print("wrote", path)


if __name__ == '__main__':
    main()
