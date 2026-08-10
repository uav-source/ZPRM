"""Direct WGS84 geodetic-to-ECEF conversion for GT-only world positions."""

from __future__ import annotations

import numpy as np


WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FIRST_ECCENTRICITY_SQUARED = 6.69437999014e-3


def wgs84_geodetic_to_ecef(
    latitude_deg: np.ndarray, longitude_deg: np.ndarray, ellipsoidal_height_m: np.ndarray
) -> np.ndarray:
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    height = np.asarray(ellipsoidal_height_m, dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.shape != height.shape:
        raise ValueError("latitude, longitude, and height shapes must match")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all() or not np.isfinite(height).all():
        raise ValueError("geodetic coordinates must be finite")
    if np.any(np.abs(latitude) > np.pi / 2.0):
        raise ValueError("latitude is outside WGS84 range")
    prime_vertical = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * np.sin(latitude) ** 2
    )
    return np.ascontiguousarray(
        np.column_stack(
            (
                (prime_vertical + height) * np.cos(latitude) * np.cos(longitude),
                (prime_vertical + height) * np.cos(latitude) * np.sin(longitude),
                (
                    prime_vertical * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)
                    + height
                )
                * np.sin(latitude),
            )
        ),
        dtype="<f8",
    )
