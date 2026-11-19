"""Module for reading binary TDMS files produced by LabView"""

from __future__ import absolute_import


# Make version number available
from .version import __version_info__, __version__

# Export public objects
from .tdms import TdmsFile, TdmsGroup, TdmsChannel, DataChunk, GroupDataChunk, ChannelDataChunk
from .online import OnlineTdmsFile, OnlineTdmsError
from .writer import TdmsWriter, RootObject, GroupObject, ChannelObject
from .export.hdf_archive import (
    export_tdms_to_hdf5,
    restore_tdms_from_hdf5,
    ArchiveError,
    ArchiveFormatError,
    UnsupportedArchiveTypeError,
)
