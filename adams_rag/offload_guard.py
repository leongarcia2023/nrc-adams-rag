"""Prevent acquisition or empty database creation while data is off-device."""
from pathlib import Path


def require_local_data(path):
    for directory in Path(path).resolve().parents:
        marker = directory.with_name(directory.name + '.offload.json') if directory.name else None
        if marker and marker.is_file():
            raise RuntimeError(
                'This NRC data collection is offline because its data was archived elsewhere. '
                'Restore the archived data and remove its .offload.json marker before starting the app. '
                f'Move record: {marker}'
            )
