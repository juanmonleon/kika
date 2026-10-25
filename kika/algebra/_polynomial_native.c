#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static PyObject *legendre_values(PyObject *self, PyObject *args)
{
    PyObject *objects[3];
    Py_buffer b[3] = {{0}};
    PyObject *result = NULL;
    double *scratch = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &objects[0], &objects[1], &objects[2])) return NULL;
    for (int j = 0; j < 3; ++j) {
        int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
        if (j == 2) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(objects[j], &b[j], flags) < 0) goto done;
        if (b[j].itemsize != sizeof(double) || !b[j].format || (strcmp(b[j].format, "d") && strcmp(b[j].format, "=d") && strcmp(b[j].format, "@d")) || b[j].len % sizeof(double)) {
            PyErr_SetString(PyExc_ValueError, "native polynomial buffers must be contiguous native float64");
            goto done;
        }
    }
    if (b[0].ndim != 1 || b[0].len == 0 || b[1].len != b[2].len) {
        PyErr_SetString(PyExc_ValueError, "nonempty coefficient vector and matching point/output buffers required");
        goto done;
    }
    uintptr_t o = (uintptr_t)b[2].buf, end = o + b[2].len;
    for (int j = 0; j < 2; ++j) {
        uintptr_t p = (uintptr_t)b[j].buf;
        if (b[2].len && b[j].len && o < p + b[j].len && p < end) {
            PyErr_SetString(PyExc_ValueError, "polynomial output must not overlap input buffers");
            goto done;
        }
    }
    for (int j = 0; j < 3; ++j) {
        if ((uintptr_t)b[j].buf % sizeof(double)) { result = Py_NewRef(Py_False); goto done; }
    }
    const double *c = (const double *)b[0].buf;
    const double *x = (const double *)b[1].buf;
    double *out = (double *)b[2].buf;
    Py_ssize_t nc = b[0].len / sizeof(double), np = b[1].len / sizeof(double);
    /* Preserve NumPy's exceptional-domain handling outside this finite regime. */
    for (Py_ssize_t j = 0; j < nc; ++j) {
        if (!isfinite(c[j]) || fabs(c[j]) > 1e100) { result = Py_NewRef(Py_False); goto done; }
    }
    for (Py_ssize_t i = 0; i < np; ++i) {
        if (!isfinite(x[i]) || fabs(x[i]) > 1.) { result = Py_NewRef(Py_False); goto done; }
    }
    if (np && nc > 2) {
        scratch = PyMem_Malloc((size_t)b[1].len);
        if (!scratch) { PyErr_NoMemory(); goto done; }
    }
    Py_BEGIN_ALLOW_THREADS
    if (nc <= 2) {
        double c0 = c[0], c1 = nc == 2 ? c[1] : 0.;
        for (Py_ssize_t i = 0; i < np; ++i) out[i] = c0 + c1 * x[i];
    } else {
        for (Py_ssize_t i = 0; i < np; ++i) { out[i] = c[nc-2]; scratch[i] = c[nc-1]; }
        for (Py_ssize_t j = 3; j <= nc; ++j) {
            double nd = (double)(nc-j+2);
            for (Py_ssize_t i = 0; i < np; ++i) {
                double previous = out[i], c1 = scratch[i];
                /* Keep the NumPy Clenshaw operation order; no fused operations.
                   This is legval as of NumPy 2.3 (numpy a7e6113, "speedup
                   evaluation of legval"), which forms the ratio first:
                     c0 = c[-i] - c1*((nd-1)/nd)
                     c1 = tmp + c1*x*((2*nd-1)/nd)
                   NumPy <= 2.2 multiplied then divided, (c1*(nd-1))/nd, and
                   differs from this in the last bit. */
                out[i] = c[nc-j] - c1 * ((nd-1.) / nd);
                scratch[i] = previous + (c1 * x[i]) * ((2.*nd-1.) / nd);
            }
        }
        for (Py_ssize_t i = 0; i < np; ++i) out[i] = out[i] + scratch[i] * x[i];
    }
    Py_END_ALLOW_THREADS
    result = Py_NewRef(Py_True);
 done:
    PyMem_Free(scratch);
    for (int j = 0; j < 3; ++j) if (b[j].obj) PyBuffer_Release(&b[j]);
    return result;
}

static PyMethodDef methods[] = {
    {"legendre_values", legendre_values, METH_VARARGS, "Evaluate a finite Legendre series on [-1,1] into a separate output buffer; false requests the NumPy reference path."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_polynomial_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__polynomial_native(void)
{
    PyObject *m = PyModule_Create(&module);
    if (m && PyModule_AddIntConstant(m, "API_VERSION", 1) < 0) { Py_DECREF(m); return NULL; }
    return m;
}
