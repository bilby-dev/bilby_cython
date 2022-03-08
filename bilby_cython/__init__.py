from . import geometry


def _get_version_information():
    import os
    version_file = os.path.join(os.path.dirname(__file__), '.version')
    if os.path.exists(version_file):
        return open(version_file, "r").readline().rstrip()


__version__ = _get_version_information()
