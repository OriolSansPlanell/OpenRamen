# Importer coverage

What the import engine makes of real libraries, measured by `benchmarks/import_coverage.py`.
Python 3.13.18, NumPy 2.5.3, OpenRamen 0.1.0.dev0.

## Running right without manual fixes

166 real calls to 156 NumPy functions are run as nodes and compared with calling NumPy directly: same values, same dtypes, and every output checked against the type the importer declared.

| | Correct | Share |
| --- | --- | --- |
| From signatures and docstrings alone | 165 of 166 | 99% |
| With the bundled NumPy node pack | 166 of 166 | 100% |

What introspection alone gets wrong:

- `numpy.histogramdd`: wrong-type. node 'n' (numpy.histogramdd), pin 'edges': output is declared Tuple[Union[Array[_, _], Bool, Complex, Float, Int], ...] (a type taken from its docstring) but the node returned List[Array[float64, 1]]

The cases are in `tests/numpy_cases.py`. They cover creation, shape, arithmetic, reductions, sorting, linear algebra and FFT, plus deliberately awkward functions: flags that change the number of results, `*args`, nested results.

## How much is understood statically

| Module | Functions | Imported | Inputs typed | Outputs typed | Fully typed nodes |
| --- | --- | --- | --- | --- | --- |
| `numpy` 2.5.3 | 379 | 100% | 73% | 88% | 36% |
| `numpy.linalg` 2.5.3 | 31 | 100% | 86% | 59% | 45% |
| `numpy.fft` 2.5.3 | 18 | 100% | 98% | 100% | 89% |
| `numpy.random` 2.5.3 | 51 | 100% | 87% | 84% | 69% |
| `scipy.ndimage` 1.18.1 | 75 | 100% | 91% | 91% | 43% |
| `scipy.signal` 1.18.1 | 149 | 100% | 93% | 86% | 68% |
| `scipy.optimize` 1.18.1 | 57 | 100% | 84% | 60% | 21% |
| `skimage.filters` 0.26.0 | 46 | 100% | 82% | 70% | 48% |
| `skimage.measure` 0.26.0 | 31 | 100% | 82% | 71% | 35% |
| `skimage.morphology` 0.26.0 | 43 | 100% | 83% | 61% | 26% |
| `skimage.transform` 0.26.0 | 29 | 100% | 84% | 81% | 38% |
| `math` | 57 | 98% | 4% | 0% | 0% |

"Typed" means the pin has a type other than `Any`. An untyped pin still works: its wires are checked when the graph runs instead of when it is drawn.

Not imported:

- `math.log`: no signature could be found

Classes and their methods are not imported yet; only functions are counted.
