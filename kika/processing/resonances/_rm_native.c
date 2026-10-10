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

#define RM_SKIP_ENVELOPE 1
#define RM_SKIP_ABSORPTION 1
#define RM_TARGET
#define RM_MATRICES matrix_sum_certified
#include "_rm_kernel.h"
#undef RM_TARGET
#undef RM_MATRICES
#undef RM_SKIP_ENVELOPE
#undef RM_SKIP_ABSORPTION

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define RM_HAS_AVX2 1
#define RM_TARGET __attribute__((target("avx2")))
#define RM_MATRICES matrix_sum_avx2
#define RM_ABSORPTION absorption_sum_avx2
#include "_rm_kernel.h"
#undef RM_TARGET
#undef RM_MATRICES
#undef RM_ABSORPTION
#define RM_SKIP_ENVELOPE 1
#define RM_SKIP_ABSORPTION 1
#define RM_TARGET __attribute__((target("avx2")))
#define RM_MATRICES matrix_sum_certified_avx2
#include "_rm_kernel.h"
#undef RM_TARGET
#undef RM_MATRICES
#undef RM_SKIP_ENVELOPE
#undef RM_SKIP_ABSORPTION
#endif
static int use_avx2=0;

static int envelope_bounded(Py_ssize_t ne,Py_ssize_t n,Py_ssize_t c,
        const double *gamma,const double *a,const double *f) {
    /* |Re(1/d)|+|Im(1/d)| <= 3/gamma. Outward rounding and
     * contraction margin certify the original per-energy envelope guard.
     * Dispatch to separate kernels so an unavailable certificate does not
     * add branches inside the existing generic inner loops.
     */
    double max_f2[3]={0},bound=0.;
    for (Py_ssize_t i=0;i<ne;i++) for (Py_ssize_t j=0;j<c;j++) {
        double value=f[i*c+j]*f[i*c+j];
        if (!isfinite(value)) return 0;
        if (value>max_f2[j]) max_f2[j]=value;
    }
    for (Py_ssize_t k=0;k<n;k++) {
        if (!(gamma[k]>0. && isfinite(gamma[k]))) return 0;
        for (Py_ssize_t j=0;j<c;j++) {
            double square=nextafter(a[k*c+j]*a[k*c+j],INFINITY);
            double product=nextafter(square*nextafter(max_f2[j],INFINITY),INFINITY);
            double term=nextafter(nextafter(3*product,INFINITY)/gamma[k],INFINITY);
            bound=nextafter(bound+term,INFINITY);
        }
        if (!(bound<=ENVELOPE_LIMIT)) return 0;
    }
    return nextafter(bound*(1+64*((double)n+1)*DBL_EPSILON),INFINITY)<=ENVELOPE_LIMIT;
}

static PyObject *matrices(PyObject *self,PyObject *args) {
    Py_ssize_t ne,n,c;PyObject *o[7];Py_buffer b[7]={0};
    if (!PyArg_ParseTuple(args,"nnnOOOOOOO",&ne,&n,&c,&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6])) return NULL;
    if (!dimensions(ne,n,c)) return NULL;
    Py_ssize_t sizes[7]={ne,n,n,n*c,ne*c,2*ne*c*c,ne};
    for (int i=0;i<7;i++) if (!buffer(o[i],&b[i],sizes[i],i==6?"B":"d",i>=5)) {release(b);return NULL;}
    Py_BEGIN_ALLOW_THREADS
    int certified=envelope_bounded(ne,n,c,b[2].buf,b[3].buf,b[4].buf);
#ifdef RM_HAS_AVX2
    if (use_avx2) {
        if (certified) matrix_sum_certified_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
        else matrix_sum_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    }
    else
#endif
    if (certified) matrix_sum_certified(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
    else matrix_sum(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf);
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

/* Constant-radius, no-competition BW: every level in source order.
 * Parameters: Er, Gn, Gg, Gf, gJ, P(Er), spin-sector index.
 * Energy rows: E, P(E), beta, sin(phi)^2, sin(2 phi).
 * Output rows retain capture/fission/SLBW accumulators, followed by the
 * two MLBW amplitudes per spin sector. No clipping or resonance window.
 */
static PyObject *breit_wigner(PyObject *self,PyObject *args) {
    Py_ssize_t ne,n,ns,l,ml;double coefficient;PyObject *o[3];Py_buffer b[7]={0};
    if (!PyArg_ParseTuple(args,"nnnnndOOO",&ne,&n,&ns,&l,&ml,&coefficient,&o[0],&o[1],&o[2])) return NULL;
    if (ne<0 || n<0 || ns<1 || ns>128 || l<0 || l>2 || (ml!=0 && ml!=1) ||
        !isfinite(coefficient) || coefficient<=0 || ne>PY_SSIZE_T_MAX/(8*(3+2*ns)) || n>PY_SSIZE_T_MAX/56) {
        PyErr_SetString(PyExc_ValueError,"invalid native BW dimensions");return NULL;
    }
    Py_ssize_t sizes[3]={ne*5,n*7,ne*(3+2*ns)};
    for (int i=0;i<3;i++) if (!buffer(o[i],&b[i],sizes[i],"d",i==2)) {release(b);return NULL;}
    double *en=b[0].buf,*levels=b[1].buf,*out=b[2].buf;int unsafe=0;
    for (Py_ssize_t k=0;k<n;k++) if (!(levels[7*k+5]>0) || !isfinite(levels[7*k+6]) ||
        levels[7*k+6]<0 || levels[7*k+6]>=ns || levels[7*k+6]!=(Py_ssize_t)levels[7*k+6]) {
        release(b);PyErr_SetString(PyExc_ValueError,"invalid native BW reference/sector");return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t k=0;k<n;k++) {
        const double *v=levels+7*k;Py_ssize_t sector=(Py_ssize_t)v[6];
        double zr=coefficient*fabs(v[0]);
        for (Py_ssize_t i=0;i<ne;i++) {
            const double *q=en+5*i;double *row=out+(3+2*ns)*i;
            double gn=v[1]*q[1]/v[5],width=gn+v[2]+v[3]+0.,delta=q[0]-v[0];
            if (l) {
                double z=coefficient*q[0],dz=coefficient*(fabs(v[0])-q[0]);
                double shift=l==1?dz/(1+zr)/(1+z):3*dz*(zr*z+6*(zr+z)+9)/((zr*zr+3*zr+9)*(z*z+3*z+9));
                delta-=v[1]*shift/(2*v[5]);
            }
            double den=delta*delta+(width/2)*(width/2),factor=q[2]*v[4];
            if (!(den>=DBL_MIN && isfinite(den)) || (v[1]!=0 && gn==0)) {unsafe=1;continue;}
            row[0]+=factor*gn*v[2]/den;row[1]+=factor*gn*v[3]/den;
            if (ml) {row[3+2*sector]+=gn*width/2/den;row[4+2*sector]+=gn*delta/den;}
            else row[2]+=factor*gn*(gn-2*width*q[3]+2*delta*q[4])/den;
        }
    }
    Py_END_ALLOW_THREADS
    release(b);return PyBool_FromLong(!unsafe);
}

static double softplus(double t) {
    return fmax(t,0.)+log1p(exp(-fabs(t)));
}

/* Same log-energy Laplace integrand as fluctuations.py. QUADPACK retains
 * the adaptive nodes, tolerances and error estimates; only the callback
 * arithmetic avoids crossing into Python at every quadrature node.
 */
static double urr_integrand(double s,void *context) {
    const double *v=(const double *)context;Py_ssize_t n=(Py_ssize_t)v[0];
    double deterministic=isinf(v[1])?0.:exp(fmin(700.,s+v[1]));
    double sum=0.;
    for (Py_ssize_t k=0;k<n;k++) sum+=v[4+2*k]*softplus(s+v[5+2*k]);
    return exp(s-deterministic-sum-softplus(s+v[2])-softplus(s+v[3]));
}

static PyMethodDef methods[]={
    {"breit_wigner",breit_wigner,METH_VARARGS,"Accumulate all constant-radius BW levels in source order."},
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
    PyObject *callback=PyCapsule_New((void *)urr_integrand,"double (double, void *)",NULL);
    if (callback==NULL || PyModule_AddObject(m,"URR_INTEGRAND",callback)<0) {
        Py_XDECREF(callback);Py_DECREF(m);return NULL;
    }
    if (PyModule_AddStringConstant(m,"SIMD_BACKEND",use_avx2?"avx2":"baseline")<0) {Py_DECREF(m);return NULL;}
    return m;
}
