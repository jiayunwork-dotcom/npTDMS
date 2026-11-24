"""Acceptance tests for windowed, storage-chunk-independent TDMS reading.

Covers:
  * Point-for-point equivalence between interleaved and contiguous files
    holding identical data, through TdmsFile.data_chunks,
    TdmsChannel.data_chunks, integer indexing, read_data slices and
    TdmsFile.read(memmap_dir=...).
  * Streaming chunk offsets are contiguous and sum to the channel length.
  * tracemalloc peak memory stays under 4 MiB for the streaming and
    memory-mapped whole-file entry points.
  * Actual bytes read (via a counting file object) stay under 256 KiB for
    single-index and 1,000-row slice reads on the large-chunk files.
  * No more than 500 chunks are yielded for the 2-rows-per-unit files.
  * Incomplete final storage chunks are handled for both layouts.
"""

import argparse
import gc
import os
import sys
import tempfile
import tracemalloc

import numpy as np

# In isolated mode (-I) the script directory is not added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')

from nptdms import TdmsFile
from nptdms.base_segment import DEFAULT_READ_WINDOW_SIZE

from counting_file import CountingFile
from generate_tdms import make_data


MEMORY_LIMIT = 4 * 1024 * 1024          # 4 MiB
BYTES_LIMIT = 256 * 1024                # 256 KiB
MAX_SMALL_FILE_CHUNKS = 500
SLICE_LENGTH = 1000

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print(f"  [FAIL] {message}")


def section(title):
    print(f"\n== {title} ==")


def channel_path_str(channel):
    return str(channel._path)


def read_via_file_data_chunks(path):
    """Stream the whole file via TdmsFile.data_chunks and verify chunk offsets."""
    result = {}
    offsets = {}
    with TdmsFile.open(path) as tdms_file:
        for chunk in tdms_file.data_chunks():
            for group in chunk.groups():
                for channel_chunk in group.channels():
                    path_key = str(channel_chunk._path)
                    previous = offsets.get(path_key, 0)
                    check(channel_chunk.offset == previous,
                          f"{os.path.basename(path)}: non-contiguous offset for {path_key}")
                    offsets[path_key] = previous + len(channel_chunk)
                    result.setdefault(path_key, []).append(np.array(channel_chunk[:]))
    arrays = {path_key: np.concatenate(parts) for path_key, parts in result.items()}
    return arrays, offsets


def read_via_channel_data_chunks(path):
    """Stream each channel via TdmsChannel.data_chunks and verify offsets."""
    result = {}
    chunk_counts = {}
    with TdmsFile.open(path) as tdms_file:
        for group in tdms_file.groups():
            for channel in group.channels():
                parts = []
                count = 0
                running_offset = 0
                for chunk in channel.data_chunks():
                    check(chunk.offset == running_offset,
                          f"{os.path.basename(path)}: non-contiguous channel offset for {channel.name}")
                    running_offset += len(chunk)
                    parts.append(np.array(chunk[:]))
                    count += 1
                result[channel_path_str(channel)] = (
                    np.concatenate(parts) if parts else np.array([]))
                chunk_counts[channel_path_str(channel)] = count
    return result, chunk_counts


def verify_layout_equivalence(prefix, expected_length):
    section(f"Correctness: {prefix}")
    contiguous = os.path.join(DATA_DIR, f"{prefix}_contiguous.tdms")
    interleaved = os.path.join(DATA_DIR, f"{prefix}_interleaved.tdms")

    file_arrays_c, file_offsets_c = read_via_file_data_chunks(contiguous)
    file_arrays_i, file_offsets_i = read_via_file_data_chunks(interleaved)
    channel_arrays_c, _counts_c = read_via_channel_data_chunks(contiguous)
    channel_arrays_i, _counts_i = read_via_channel_data_chunks(interleaved)

    channels, expected = make_data(expected_length)

    for path in channels:
        check(file_offsets_c[path] == expected_length,
              f"{prefix}: contiguous file offsets sum to channel length")
        check(file_offsets_i[path] == expected_length,
              f"{prefix}: interleaved file offsets sum to channel length")
        check(np.array_equal(file_arrays_c[path], expected[path]),
              f"{prefix}: contiguous data_chunks matches expected {path}")
        check(np.array_equal(file_arrays_i[path], expected[path]),
              f"{prefix}: interleaved data_chunks matches expected {path}")
        check(np.array_equal(channel_arrays_c[path], expected[path]),
              f"{prefix}: contiguous channel data_chunks matches expected {path}")
        check(np.array_equal(channel_arrays_i[path], expected[path]),
              f"{prefix}: interleaved channel data_chunks matches expected {path}")
        check(np.array_equal(file_arrays_c[path], file_arrays_i[path]),
              f"{prefix}: file data_chunks layouts match {path}")
        check(np.array_equal(channel_arrays_c[path], channel_arrays_i[path]),
              f"{prefix}: channel data_chunks layouts match {path}")

    with TdmsFile.open(contiguous) as tf_c, TdmsFile.open(interleaved) as tf_i:
        for path in channels:
            ch_c = next(c for g in tf_c.groups() for c in g.channels() if channel_path_str(c) == path)
            ch_i = next(c for g in tf_i.groups() for c in g.channels() if channel_path_str(c) == path)
            for index in (0, expected_length // 2, expected_length - 1):
                check(ch_c[index] == expected[path][index] and ch_i[index] == expected[path][index],
                      f"{prefix}: index {index} matches expected {path}")
                check(ch_c[index] == ch_i[index],
                      f"{prefix}: layouts agree at index {index} {path}")
            start = expected_length // 2
            check(np.array_equal(ch_c[start:start + SLICE_LENGTH], expected[path][start:start + SLICE_LENGTH]),
                  f"{prefix}: contiguous slice matches expected {path}")
            check(np.array_equal(ch_i[start:start + SLICE_LENGTH], expected[path][start:start + SLICE_LENGTH]),
                  f"{prefix}: interleaved slice matches expected {path}")
            check(np.array_equal(ch_c.read_data(start, SLICE_LENGTH),
                                 expected[path][start:start + SLICE_LENGTH]),
                  f"{prefix}: contiguous read_data matches expected {path}")
            check(np.array_equal(ch_i.read_data(start, SLICE_LENGTH),
                                 expected[path][start:start + SLICE_LENGTH]),
                  f"{prefix}: interleaved read_data matches expected {path}")

    with tempfile.TemporaryDirectory() as memmap_dir:
        tf_c = TdmsFile.read(contiguous, memmap_dir=memmap_dir)
        tf_i = TdmsFile.read(interleaved, memmap_dir=memmap_dir)
        for path in channels:
            ch_c = next(c for g in tf_c.groups() for c in g.channels() if channel_path_str(c) == path)
            ch_i = next(c for g in tf_i.groups() for c in g.channels() if channel_path_str(c) == path)
            check(np.array_equal(np.array(ch_c.data), expected[path]),
                  f"{prefix}: contiguous memmap read matches expected {path}")
            check(np.array_equal(np.array(ch_i.data), expected[path]),
                  f"{prefix}: interleaved memmap read matches expected {path}")
            check(np.array_equal(ch_c.data, ch_i.data),
                  f"{prefix}: memmap layouts match {path}")
    print(f"  ({prefix} equivalence checks complete)")


def verify_truncated():
    section("Correctness: incomplete final storage chunk")
    base_rows = 200_000
    num_units = 4
    truncated_rows = 123
    total_rows = base_rows * (num_units - 1) + truncated_rows
    contiguous = os.path.join(DATA_DIR, "truncated_contiguous.tdms")
    interleaved = os.path.join(DATA_DIR, "truncated_interleaved.tdms")

    arrays_c, offsets_c = read_via_file_data_chunks(contiguous)
    arrays_i, offsets_i = read_via_file_data_chunks(interleaved)
    channels, expected = make_data(total_rows)

    for path in channels:
        check(offsets_c[path] == total_rows and offsets_i[path] == total_rows,
              "truncated: offsets sum to channel length")
        check(np.array_equal(arrays_c[path], expected[path]),
              f"truncated contiguous matches expected {path}")
        check(np.array_equal(arrays_i[path], expected[path]),
              f"truncated interleaved matches expected {path}")
        check(np.array_equal(arrays_c[path], arrays_i[path]),
              f"truncated: layouts match {path}")
    print("  (truncated chunk checks complete)")


def measure_tracemalloc_peak(operation):
    gc.collect()
    tracemalloc.start(25)
    operation()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return peak


def verify_memory(prefix):
    section(f"Memory (tracemalloc peak < 4 MiB): {prefix}")
    for layout in ("contiguous", "interleaved"):
        path = os.path.join(DATA_DIR, f"{prefix}_{layout}.tdms")

        def iterate_file_chunks():
            with TdmsFile.open(path) as tdms_file:
                for chunk in tdms_file.data_chunks():
                    for group in chunk.groups():
                        for channel_chunk in group.channels():
                            _ = channel_chunk[:]

        def iterate_channel_chunks():
            with TdmsFile.open(path) as tdms_file:
                for group in tdms_file.groups():
                    for channel in group.channels():
                        for chunk in channel.data_chunks():
                            _ = chunk[:]

        def iterate_channel_values():
            with TdmsFile.open(path) as tdms_file:
                for group in tdms_file.groups():
                    for channel in group.channels():
                        for _value in channel:
                            pass

        def full_memmap_read():
            with tempfile.TemporaryDirectory() as memmap_dir:
                tdms_file = TdmsFile.read(path, memmap_dir=memmap_dir)
                for group in tdms_file.groups():
                    for channel in group.channels():
                        _ = channel.data.sum()

        peaks = {
            "TdmsFile.data_chunks": measure_tracemalloc_peak(iterate_file_chunks),
            "TdmsChannel.data_chunks": measure_tracemalloc_peak(iterate_channel_chunks),
            "channel iteration": measure_tracemalloc_peak(iterate_channel_values),
            "TdmsFile.read(memmap_dir)": measure_tracemalloc_peak(full_memmap_read),
        }
        for name, peak in peaks.items():
            check(peak < MEMORY_LIMIT,
                  f"{prefix} {layout} {name}: peak {peak / 1024:.1f} KiB exceeds 4 MiB")
            print(f"  {prefix} {layout} {name}: peak {peak / 1024:.1f} KiB")


def verify_bytes_read():
    section("Bytes read (< 256 KiB per operation): large-chunk files")
    length = 800_000
    boundary = 200_000
    indices = {"mid": length // 2, "end": length - 1}
    slice_starts = {
        "mid": length // 2,
        "cross_chunk_boundary": boundary - SLICE_LENGTH // 2,
    }

    for layout in ("contiguous", "interleaved"):
        path = os.path.join(DATA_DIR, f"large_{layout}.tdms")
        for label, index in indices.items():
            counting_file = CountingFile(path)
            with TdmsFile.open(counting_file) as tdms_file:
                counting_file.reset_counter()
                channel = tdms_file.groups()[0].channels()[0]
                _ = channel[index]
                bytes_read = counting_file.bytes_read
            check(bytes_read < BYTES_LIMIT,
                  f"large {layout} index {label}: {bytes_read / 1024:.1f} KiB exceeds 256 KiB")
            print(f"  large {layout} index {label} ({index}): {bytes_read / 1024:.1f} KiB")

        for label, start in slice_starts.items():
            counting_file = CountingFile(path)
            with TdmsFile.open(counting_file) as tdms_file:
                counting_file.reset_counter()
                channel = tdms_file.groups()[0].channels()[0]
                _ = channel[start:start + SLICE_LENGTH]
                bytes_read = counting_file.bytes_read
            check(bytes_read < BYTES_LIMIT,
                  f"large {layout} slice {label}: {bytes_read / 1024:.1f} KiB exceeds 256 KiB")
            print(f"  large {layout} slice {label}: {bytes_read / 1024:.1f} KiB")


def verify_small_file_chunk_count():
    section("Chunk count (<= 500): 2-rows-per-unit files")
    for layout in ("contiguous", "interleaved"):
        path = os.path.join(DATA_DIR, f"small_{layout}.tdms")
        file_count = 0
        channel_counts = []
        with TdmsFile.open(path) as tdms_file:
            for _chunk in tdms_file.data_chunks():
                file_count += 1
            for group in tdms_file.groups():
                for channel in group.channels():
                    channel_counts.append(sum(1 for _ in channel.data_chunks()))
        check(file_count <= MAX_SMALL_FILE_CHUNKS,
              f"small {layout} file data_chunks produced {file_count} chunks")
        check(all(count <= MAX_SMALL_FILE_CHUNKS for count in channel_counts),
              f"small {layout} channel data_chunks produced {channel_counts}")
        print(f"  small {layout}: {file_count} file chunks, channel chunks {channel_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()

    print(f"DEFAULT_READ_WINDOW_SIZE = {DEFAULT_READ_WINDOW_SIZE} values "
          f"({DEFAULT_READ_WINDOW_SIZE * 8 / 1024:.0f} KiB per float64 channel window)")

    verify_layout_equivalence("large", 800_000)
    verify_layout_equivalence("small", 200_000)
    verify_truncated()
    if not args.skip_memory:
        verify_memory("large")
        verify_memory("small")
    verify_bytes_read()
    verify_small_file_chunk_count()

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for failure in failures:
            print("  -", failure)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
