"""VisionIQ - intelligent CCTV monitoring.

A real pipeline that replaces the original static HTML mock-up:

    frame  ->  motion detection  ->  tracking  ->  per-zone features
           ->  sequence autoencoder  ->  anomaly score
           ->  access-control correlation  ->  events  ->  console / SQLite

The package is import-safe without optional heavy dependencies; the
:mod:`visioniq.vision.anomaly` module falls back to a NumPy autoencoder when
PyTorch is not installed.
"""

from __future__ import annotations

__version__ = "3.0.0"
__all__ = ["__version__"]
