"""Utilities for the UC Irvine course on 'ML & Statistics for Physicists'
"""

import os.path


def locate_data(name, check_exists=True):
    """Locate the named data file.

    Data files under mls/data/ are copied when this package is installed.
    This function locates these files relative to the install directory.

    Parameters
    ----------
    name : str
        Path of data file relative to mls/data.
    check_exists : bool
        Raise a RuntimeError if the named file does not exist when this is True.

    Returns
    -------
    str
        Path of data file within installation directory.
    """
    import mls
    pkg_path = mls.__path__[0]
    path = os.path.join(pkg_path, 'data', name)
    if check_exists and not os.path.exists(path):
        raise RuntimeError('No such data file: {}'.format(path))
    return path
