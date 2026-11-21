# Mathematics of tabulated data

Almost every quantity in an evaluation is a tabulated function: pairs of points joined by
an interpolation law. This chapter describes the operations kika performs on such
functions, which every other method rests on: evaluating at any abscissa, integrating
exactly under each interpolation law, averaging over groups, linearising, adding two
functions on different grids, and folding with a Gaussian.

Where this code sits in the library, and why it imports nothing else from kika, is in
Part I ({doc}`../../architecture/algebra-layer`). The chapter has one page.

- {doc}`tabulated-functions`: the interpolation laws, exact integrals and group
  averages, linearisation, sums and the Gaussian fold.

```{toctree}
:hidden:

tabulated-functions
```
