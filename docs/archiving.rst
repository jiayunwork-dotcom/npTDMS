Lossless HDF5 archives
======================

The regular :py:meth:`~nptdms.TdmsFile.as_hdf` export is intended for getting
data out of TDMS into HDF5 for analysis: it writes *scaled* sample values and
renders timestamps as microsecond-resolution text, so it cannot reconstruct the
original TDMS file.  For long term storage where the TDMS file may one day need
to be handed back, npTDMS provides a separate, reversible archive format with
two entry points:

* :py:meth:`~nptdms.TdmsFile.write_hdf5_archive` (or
  :func:`~nptdms.export_tdms_to_hdf5`) writes a new HDF5 archive from a
  :class:`~nptdms.TdmsFile`.
* :py:meth:`~nptdms.TdmsFile.restore_hdf5_archive` (or
  :func:`~nptdms.restore_tdms_from_hdf5`) rebuilds a new TDMS file from an
  archive, without needing the original TDMS file and without embedding a copy
  of the original TDMS bytes.

Basic usage::

    from nptdms import TdmsFile

    # Archive a TDMS file.  Both TdmsFile.read and TdmsFile.open work; opening
    # the file streams data from disk one chunk at a time for long channels.
    with TdmsFile.open("acquisition.tdms", raw_timestamps=True) as tdms_file:
        tdms_file.write_hdf5_archive("acquisition.h5")

    # Rebuild the TDMS file later, with at most 100 000 samples per batch.
    TdmsFile.restore_hdf5_archive(
        "acquisition.h5", "acquisition_restored.tdms", batch_size=100000)

Both target paths must refer to new files; existing files are never
overwritten, and output is written to a temporary file in the same directory
and moved into place only when complete, so a failure never leaves a partial
file behind.

What is preserved
-----------------

* **Raw, unscaled samples.**  Integer channels keep their integer values,
  dtype and order.  Linear (and other) scaling lives in the channel properties,
  which are archived verbatim, so ``read_data(scaled=False)`` returns the
  original integers and normal scaled reads still match the source.
* **Full timestamp precision.**  Timestamps are stored as a compound dataset
  of signed 64 bit seconds and unsigned 64 bit second fractions (2**-64
  seconds).  Two timestamps differing by a single fraction bit remain
  distinct.  Compare restored timestamps with
  ``TdmsFile.read(..., raw_timestamps=True)``.
* **All supported TDMS data types.**  Booleans, signed and unsigned integers,
  float32/float64 (including the "with unit" variants), strings and
  timestamps.  Strings keep empty values, non-ASCII characters and embedded
  NUL bytes.
* **Object metadata.**  Root, group and channel properties, the TDMS format
  version, empty groups and empty channels, and each channel's own length.
  Group and channel names containing slashes, quotes or spaces are preserved.

DAQmx raw data, complex numbers and other unsupported types are rejected up
front with an :class:`~nptdms.UnsupportedArchiveTypeError` naming the channel
that cannot be archived; no partial archive is written.

Archive layout (format version 1)
---------------------------------

The archive is an ordinary HDF5 file that can be browsed and sliced directly
with h5py::

    /
    |-- .tdms_archive_meta/
    |     |-- format                  "nptdms-tdms-archive"
    |     |-- format_version          integer format version (currently 1)
    |     |-- tdms_version            TDMS version (4712 or 4713)
    |     |-- manifest                JSON dataset listing every object and
    |     |                           its exact TDMS type and length
    |     `-- .tdms_nul_properties/   byte datasets for string properties that
    |                                 contain NUL bytes, keyed by object
    |-- <group>/<channel>/data        Raw samples for each channel
    `-- TDMS properties               Stored as HDF5 attributes on the root,
                                      group and channel objects

TDMS names are mapped to HDF5 link names with a reversible escaping rule
(``%`` -> ``%25``, ``/`` -> ``%2F``, empty name -> ``%00``), and the original
names are also kept in the ``.tdms_archive_name`` attribute and the manifest,
so names with slashes or quotes are never split or mangled.  HDF5 string
attributes cannot contain NUL bytes, so the rare string properties that do are
stored as variable length byte datasets under ``.tdms_nul_properties`` instead.

Channel datasets are chunked and use:

* the exact raw NumPy dtype for numeric and boolean channels,
* a ``(second_fractions: uint64, seconds: int64)`` compound dtype for
  timestamps,
* variable length uint8 arrays for strings, storing raw UTF-8 bytes so NUL
  bytes and empty strings survive.

Because each channel is one contiguous, chunked dataset, a single channel can
be read or sliced independently, for example
``h5py.File(...)["climate"]["temperature"]["data"][100:200]``.  Timestamps can
be sliced field by field, such as ``dataset[start:stop]["seconds"]``.

Streaming and memory use
------------------------

Data is streamed in both directions rather than accumulated:

* When archiving an opened (not fully read) TDMS file, the existing TDMS
  chunked reader hands each decoded source chunk straight to the HDF5 export
  layer.  At most one source decode chunk of samples is buffered, and the
  reader's per-channel segment index used for normal slicing is retained.
* When restoring, ``batch_size`` caps the number of samples read from HDF5 and
  handed to the existing TDMS writer path in one go.  Each batch is released
  after it is written.

Neither side ever holds a buffer the size of a whole long channel.  The source
and destination may use completely different chunking: TDMS segments and HDF5
chunks are independent of each other.

Error handling
--------------

* An existing destination path raises :class:`FileExistsError`.
* A truncated source TDMS file raises :class:`~nptdms.ArchiveError`.
* An unrecognised archive, an unknown format version, missing channel data, or
  a dataset type or length that disagrees with the manifest raises
  :class:`~nptdms.ArchiveFormatError`.

In every failure case the destination path is left without any partial file.
