pyblock
=======

``pyblock`` is a python module for performing a reblocking analysis on
serially-correlated data.

The algorithms implemented in ``pyblock`` are not new; please see the documentation for
references.

``pyblock`` requires Python >= 3.10 and is compatible with modern versions of ``numpy`` and ``pandas``.

.. image:: https://img.shields.io/pypi/v/pyblock.svg
        :target: https://pypi.python.org/pypi/pyblock

.. image:: https://github.com/jsspencer/pyblock/workflows/Test%20pyblock/badge.svg?branch=master
        :target: https://github.com/jsspencer/pyblock/actions?query=workflow%3A%22Test+pyblock%22+branch%3Amaster
        :alt: CI build status
        
.. image:: https://codecov.io/gh/jsspencer/pyblock/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jsspencer/pyblock

.. image:: https://readthedocs.org/projects/pyblock/badge/?version=latest
        :target: https://pyblock.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Documentation
-------------

Documentation and a simple tutorial can be found in the docs subdirectory and on
`readthedocs <https://pyblock.readthedocs.io>`_.

Installation
------------

``pyblock`` can be installed from PyPI:

::

    $ pip install pyblock

To install from source:

::

    $ pip install /path/to/pyblock

To install an editable version (useful for development work):

::

    $ pip install -e ".[test,opts]"

``pyblock`` requires ``numpy`` and (optionally) ``pandas`` and ``matplotlib``. Please see the
documentation for more details.

License
-------

Modified BSD license; see LICENSE for more details.

Please cite ``pyblock, James Spencer, http://github.com/jsspencer/pyblock`` if used to
analyse data for an academic publication.

Author
------

James Spencer, Imperial College London

Contributing
------------

Contributions are extremely welcome, either by raising an issue or contributing code.
For code contributions, please try to follow the following points:

#. Divide commits into logical units (e.g. don't mix feature development with
   refactoring).
#. Ensure all existing tests pass.
#. Create tests for new functionality. I aim for complete test coverage.
#. Write nice git commit messages (see `Tim Pope's advice <http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_.)
#. Send a pull request!

Acknowledgments
---------------

Will Vigor (Imperial College London) pointed out and wrote an early implementation of
the algorithm to detect the optimal reblock length.

Tom Poole (Imperial College London) contributed code to handle weighted averages.

The HANDE FCIQMC/CCMC development team made several helpful comments and suggestions.
