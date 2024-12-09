# API

## I/O

```{eval-rst}
.. module:: fishtank.io
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    io.read_img
    io.read_fov
    io.read_mosaic
    io.list_fovs
    io.read_dax
    io.read_xml
    io.read_color_usage

```

## Plotting

```{eval-rst}
.. module:: fishtank.pl
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    pl.imshow
```

## Correction

```{eval-rst}
.. module:: fishtank.correct
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    correct.illumination
    correct.color_normalization
    correct.spot_alignment

```

## Filters

```{eval-rst}
.. module:: fishtank.filters
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    filters.deconwolf
    filters.unsharp_mask

```

## Decode

```{eval-rst}
.. module:: fishtank.decode
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    decode.spot_intensities
    decode.expectation_maximization
    decode.logistic_regression

```

## Segmentation

```{eval-rst}
.. module:: fishtank.seg
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    seg.masks_to_polygons
    seg.polygons_to_masks
    seg.fix_overlaps
    seg.polygon_properties
    seg.assign_spots

```

## Utilities

```{eval-rst}
.. module:: fishtank.utils
.. currentmodule:: fishtank

.. autosummary::
    :toctree: generated

    utils.tile_polygons
    utils.tile_image
    utils.create_mosaic
    utils.determine_fov_format

```
