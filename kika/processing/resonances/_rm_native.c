/* All-level RM sums, with bounded stack tiles and no NumPy C ABI.
 * Float64 only: no fast-math, artificial widths or resonance windows.
 * Rows flagged by matrices must be solved by the Python reference kernel.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __FAST_MATH__
#error "RM requires strict floating-point arithmetic"
#endif
typedef char rm_requires_ieee_float64[
    (sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024)?1:-1];

#define TILE 128
#define ENVELOPE_LIMIT 1e8

static int dimensions(Py_ssize_t ne, Py_ssize_t n, Py_ssize_t c) {
    if (ne < 0 || n < 0 || c < 1 || c > 3 ||
        ne > PY_SSIZE_T_MAX / (16*c*c) || n > PY_SSIZE_T_MAX / (8*c)) {
        PyErr_SetString(PyExc_ValueError, "invalid native RM dimensions");
        return 0;
    }
    return 1;
}

static int buffer(PyObject *obj, Py_buffer *b, Py_ssize_t count,
                  const char *format, int writable) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj,b,flags) < 0) return 0;
    Py_ssize_t item = strcmp(format,"d") == 0 ? 8 : 1;
    if (((uintptr_t)b->buf % (uintptr_t)item) != 0 || b->ndim != 1 || b->itemsize != item || b->format == NULL ||
        strcmp(b->format,format) != 0 || b->len != count*item) {
        PyErr_SetString(PyExc_ValueError,"native RM requires exact contiguous float64/uint8 buffers");
        return 0;
    }
    return 1;
}

static void release(Py_buffer *b) {
    for (int i=0;i<7;i++) if (b[i].obj != NULL) PyBuffer_Release(&b[i]);
}

#define RM_TARGET
#define RM_MATRICES matrix_sum
#define RM_ABSORPTION absorption_sum
#include "_rm_kernel.h"
#undef RM_TARGET
#undef RM_MATRICES
#undef RM_ABSORPTION

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define RM_HAS_AVX2 1
#define RM_TARGET __attribute__((target("avx2")))
#define RM_MATRICES matrix_sum_avx2
#define RM_ABSORPTION absorption_sum_avx2
#include "_rm_kernel.h"
#undef RM_TARGET
#undef RM_MATRICES
#undef RM_ABSORPTION
#endif
static int use_avx2=0;

static PyObject *matrices(PyObject *self,PyObject *args) {
    Py_ssize_t ne,n,c;PyObject *o[7];Py_buffer b[7]={0};
    if (!PyArg_ParseTuple(args,"nnnOOOOOOO",&ne,&n,&c,&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6])) return NULL;
    if (!dimensions(ne,n,c)) return NULL;
    Py_ssize_t sizes[7]={ne,n,n,n*c,ne*c,2*ne*c*c,ne};
    for (int i=0;i<7;i++) if (!buffer(o[i],&b[i],sizes[i],i==6?"B":"d",i>=5)) {release(b);return NULL;}
    Py_BEGIN_ALLOW_THREADS
#ifdef RM_HAS_AVX2
    if (use_avx2) matrix_sum_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    else
#endif
    matrix_sum(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    Py_END_ALLOW_THREADS
    release(b);Py_RETURN_NONE;
}

static PyObject *absorption(PyObject *self,PyObject *args) {
    Py_ssize_t ne,n,c;PyObject *o[7];Py_buffer b[7]={0};
    if (!PyArg_ParseTuple(args,"nnnOOOOOOO",&ne,&n,&c,&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6])) return NULL;
    if (!dimensions(ne,n,c)) return NULL;
    Py_ssize_t sizes[7]={ne,n,n,n*c,ne*c,2*ne*c,ne};
    for (int i=0;i<7;i++) if (!buffer(o[i],&b[i],sizes[i],"d",i==6)) {release(b);return NULL;}
    Py_BEGIN_ALLOW_THREADS
#ifdef RM_HAS_AVX2
    if (use_avx2) absorption_sum_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    else
#endif
    absorption_sum(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    Py_END_ALLOW_THREADS
    release(b);Py_RETURN_NONE;
}

static PyMethodDef methods[]={
    {"matrices",matrices,METH_VARARGS,"Fill all-level collision sums and reference-fallback flags."},
    {"absorption",absorption,METH_VARARGS,"Fill positive radiative probabilities using compensated level sums."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_rm_native",NULL,0,methods};
PyMODINIT_FUNC PyInit__rm_native(void) {
#ifdef RM_HAS_AVX2
    __builtin_cpu_init();
    use_avx2=__builtin_cpu_supports("avx2")!=0;
    /* For reproducible portability tests, before the module is loaded. */
    const char *baseline=Py_GETENV("KIKA_RM_BASELINE");
    if (baseline!=NULL && strcmp(baseline,"1")==0) use_avx2=0;
#endif
    PyObject *m=PyModule_Create(&module);
    if (m==NULL) return NULL;
    if (PyModule_AddIntConstant(m,"API_VERSION",1)<0) {Py_DECREF(m);return NULL;}
    if (PyModule_AddStringConstant(m,"SIMD_BACKEND",use_avx2?"avx2":"baseline")<0) {Py_DECREF(m);return NULL;}
    return m;
}
