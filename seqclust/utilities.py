"""
General utilities

"""


import os


# class NoDaemonProcess(mproc.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)


# class NDPool(multiprocessing.pool.Pool):
#     """ We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
#     because the latter is only a wrapper function, not a proper class.
#
#     >>> pool = NDPool(1)
#     """
#     Process = NoDaemonProcess


def update_path(path_file, lim_depth=5, absolute=True):
    """ bubble in the folder tree up until it found desired file
    otherwise return original one

    :param str path_file: path to the file/folder
    :param int lim_depth: length of bubble attempted
    :param bool absolute: absolute path
    :return str:

    >>> os.path.exists(update_path('README.md', absolute=False))
    True
    >>> os.path.exists(update_path('~'))
    True
    >>> os.path.exists(update_path('/'))
    True
    """
    if path_file.startswith('/'):
        return path_file
    elif path_file.startswith('~'):
        path_file = os.path.expanduser(path_file)
    else:
        for _ in range(lim_depth):
            if os.path.exists(path_file):
                break
            path_file = os.path.join('..', path_file)
    if absolute:
        path_file = os.path.abspath(path_file)
    return path_file


def sentence_tokenize(sentence, spec_chars='.,;!?'):
    """ prepossess sentence, all as lower characters and remove special chars

    :param str sentence:
    :return [str]:

    >>> s = 'Hi there, how are you?'
    >>> sentence_tokenize(s)
    ['hi', 'there', 'how', 'are', 'you']
    """
    for char in spec_chars:
        sentence = sentence.replace(char, ' ')
    tokens = sentence.lower().split()
    return tokens
