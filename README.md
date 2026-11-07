# pyblock

`pyblock` is a Python library for performing reblocking analysis on serially-correlated data.

The algorithms implemented in `pyblock` are based on established literature; please see the documentation for references.

[![PyPI version](https://img.shields.io/pypi/v/pyblock.svg)](https://pypi.org/project/pyblock/)
[![CI](https://github.com/jsspencer/pyblock/workflows/Test%20pyblock/badge.svg?branch=master)](https://github.com/jsspencer/pyblock/actions?query=workflow%3A%22Test+pyblock%22+branch%3Amaster)
[![codecov](https://codecov.io/gh/jsspencer/pyblock/branch/master/graph/badge.svg)](https://codecov.io/gh/jsspencer/pyblock)
[![Documentation Status](https://readthedocs.org/projects/pyblock/badge/?version=latest)](https://pyblock.readthedocs.io/en/latest/?badge=latest)

---

## Documentation

Documentation and a tutorial can be found in the `docs` directory and on [Read the Docs](https://pyblock.readthedocs.io).

## Installation

`pyblock` can be installed from PyPI:

```bash
pip install pyblock
```

To install from source:

```bash
pip install /path/to/pyblock
```

To install an editable version with development/optional dependencies:

```bash
pip install -e ".[test,opts]"
```

`pyblock` requires `numpy` and optionally `pandas` and `matplotlib`. Please see the documentation for details.

## License

Modified BSD license; see [LICENSE](LICENSE) for details.

Please cite `pyblock, James Spencer, http://github.com/jsspencer/pyblock` if used to analyse data for an academic publication.

## Author

James Spencer

## Contributing

Contributions are welcome, either by raising an issue or contributing code. For code contributions, please follow these guidelines:

1. Divide commits into logical units (e.g. do not mix feature development with refactoring).
2. Ensure all existing tests pass.
3. Create tests for new functionality (aim for complete test coverage).
4. Write clear commit messages (see [Tim Pope's advice](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)).
5. Open a pull request!

## Acknowledgments

- Will Vigor (Imperial College London) pointed out and wrote an early implementation of the algorithm to detect the optimal reblock length.
- Tom Poole (Imperial College London) contributed code to handle weighted averages.
- The HANDE FCIQMC/CCMC development team made several helpful comments and suggestions.
