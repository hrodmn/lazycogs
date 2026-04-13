"""Mosaic methods for combining overlapping raster tiles.

Ported from rio-tiler's ``mosaic/methods/`` (MIT licence). These are pure
numpy operations with no GDAL dependency.

All methods operate on 2-D ``numpy.ma.MaskedArray`` slices of shape
``(bands, height, width)``.  Masked pixels (``mask == True``) are treated as
no-data and filled in from subsequent tiles until the mosaic is complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.ma as ma


class MosaicMethodBase(ABC):
    """Abstract base class for pixel-selection mosaic methods."""

    def __init__(self) -> None:
        """Initialise an empty mosaic accumulator."""
        self._mosaic: ma.MaskedArray | None = None

    @property
    def is_done(self) -> bool:
        """Return ``True`` when every output pixel has a valid value."""
        if self._mosaic is None:
            return False
        return not bool(np.any(ma.getmaskarray(self._mosaic)))

    @property
    def data(self) -> np.ndarray:
        """Return the filled mosaic as a plain numpy array.

        Remaining masked pixels are filled with zero.
        """
        if self._mosaic is None:
            raise ValueError("No data has been fed to the mosaic method.")
        return ma.filled(self._mosaic, 0)

    @abstractmethod
    def feed(self, arr: ma.MaskedArray) -> None:
        """Incorporate a new tile into the mosaic.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.  Masked
                positions indicate no-data pixels in the new tile.

        """
        ...


class FirstMethod(MosaicMethodBase):
    """Use the first valid pixel encountered (first-on-top compositing)."""

    def feed(self, arr: ma.MaskedArray) -> None:
        """Incorporate ``arr`` by filling any still-empty positions.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        if self._mosaic is None:
            self._mosaic = arr.copy()
            return

        cur_mask = ma.getmaskarray(self._mosaic)
        new_mask = ma.getmaskarray(arr)

        update = cur_mask & ~new_mask
        if not np.any(update):
            return

        data = self._mosaic.data.copy()
        data[update] = arr.data[update]
        self._mosaic = ma.MaskedArray(data, mask=cur_mask & new_mask)


class HighestMethod(MosaicMethodBase):
    """Use the pixel with the highest value across all tiles."""

    def feed(self, arr: ma.MaskedArray) -> None:
        """Incorporate ``arr`` by keeping the maximum value at each position.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        if self._mosaic is None:
            self._mosaic = arr.copy()
            return

        stacked = ma.array([self._mosaic, arr])
        self._mosaic = stacked.max(axis=0)


class LowestMethod(MosaicMethodBase):
    """Use the pixel with the lowest value across all tiles."""

    def feed(self, arr: ma.MaskedArray) -> None:
        """Incorporate ``arr`` by keeping the minimum value at each position.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        if self._mosaic is None:
            self._mosaic = arr.copy()
            return

        stacked = ma.array([self._mosaic, arr])
        self._mosaic = stacked.min(axis=0)


class MeanMethod(MosaicMethodBase):
    """Use the mean of all valid pixel values across tiles."""

    def __init__(self) -> None:
        """Initialise accumulators for incremental mean computation."""
        super().__init__()
        self._count: np.ndarray | None = None

    def feed(self, arr: ma.MaskedArray) -> None:
        """Incorporate ``arr`` into the running mean.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        valid = ~ma.getmaskarray(arr)
        if self._mosaic is None:
            self._mosaic = arr.copy()
            self._count = valid.astype(np.float64)
            return

        data = self._mosaic.data.copy()
        self._count = self._count + valid  # type: ignore[operator]
        nonzero = self._count > 0
        data[nonzero] = (
            data[nonzero] * (self._count[nonzero] - valid[nonzero])
            + arr.data[nonzero] * valid[nonzero]
        ) / self._count[nonzero]
        mask = ~nonzero
        self._mosaic = ma.MaskedArray(data, mask=mask)

    @property
    def data(self) -> np.ndarray:
        """Return filled mean mosaic.

        Returns:
            Numpy array with shape ``(bands, height, width)``.

        """
        if self._mosaic is None:
            raise ValueError("No data has been fed to the mosaic method.")
        return ma.filled(self._mosaic, 0)


class MedianMethod(MosaicMethodBase):
    """Use the median of all valid pixel values across tiles."""

    def __init__(self) -> None:
        """Initialise the tile stack."""
        super().__init__()
        self._stack: list[ma.MaskedArray] = []

    def feed(self, arr: ma.MaskedArray) -> None:
        """Add ``arr`` to the stack.  Median is computed lazily in ``data``.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        self._stack.append(arr)
        # Keep _mosaic updated so is_done works correctly.
        if self._mosaic is None:
            self._mosaic = arr.copy()
        else:
            stacked = ma.array(self._stack)
            self._mosaic = ma.median(stacked, axis=0)

    @property
    def data(self) -> np.ndarray:
        """Return the pixel-wise median of all fed tiles.

        Returns:
            Numpy array with shape ``(bands, height, width)``.

        """
        if not self._stack:
            raise ValueError("No data has been fed to the mosaic method.")
        stacked = ma.array(self._stack)
        return ma.filled(ma.median(stacked, axis=0), 0)


class StdevMethod(MosaicMethodBase):
    """Use the standard deviation of all valid pixel values across tiles."""

    def __init__(self) -> None:
        """Initialise the tile stack."""
        super().__init__()
        self._stack: list[ma.MaskedArray] = []

    def feed(self, arr: ma.MaskedArray) -> None:
        """Add ``arr`` to the stack.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        self._stack.append(arr)
        if self._mosaic is None:
            self._mosaic = arr.copy()
        else:
            stacked = ma.array(self._stack)
            self._mosaic = stacked.std(axis=0)

    @property
    def data(self) -> np.ndarray:
        """Return the pixel-wise standard deviation of all fed tiles.

        Returns:
            Numpy array with shape ``(bands, height, width)``.

        """
        if not self._stack:
            raise ValueError("No data has been fed to the mosaic method.")
        stacked = ma.array(self._stack)
        return ma.filled(stacked.std(axis=0), 0)


class CountMethod(MosaicMethodBase):
    """Count the number of valid observations at each pixel."""

    def feed(self, arr: ma.MaskedArray) -> None:
        """Accumulate the count of valid pixels.

        Args:
            arr: Masked array with shape ``(bands, height, width)``.

        """
        valid = (~ma.getmaskarray(arr)).astype(np.uint16)
        if self._mosaic is None:
            self._mosaic = ma.MaskedArray(valid, mask=False)
            return

        self._mosaic = ma.MaskedArray(self._mosaic.data + valid, mask=False)

    @property
    def data(self) -> np.ndarray:
        """Return the per-pixel observation count.

        Returns:
            Numpy array with shape ``(bands, height, width)``.

        """
        if self._mosaic is None:
            raise ValueError("No data has been fed to the mosaic method.")
        return self._mosaic.data
