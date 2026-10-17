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
/* Energies per block of the level sums (_rm_kernel.h). */
#define RM_BLOCK 4
#if defined(__GNUC__) || defined(__clang__)
#define RM_INLINE static inline __attribute__((always_inline))
/* GCC/clang vector extensions: on the AVX2 target the level sums run four
 * energies per vector (RM_VECTOR in _rm_kernel.h). */
typedef double rm_v4d __attribute__((vector_size(32)));
typedef long long rm_v4i __attribute__((vector_size(32)));
#else
#define RM_INLINE static __forceinline
#endif
#define ENVELOPE_LIMIT 1e8
/* Channels per spin group. Cl-35 (ENDF/B-VIII.1) has groups of four: a
 * neutron and a proton channel for each of two channel spins. Stack tiles
 * grow as 2*RM_MAX_CHANNELS^2*TILE doubles, 74 KB at six (absorption), and
 * the level-sum blocks as 2*RM_MAX_CHANNELS^2*RM_BLOCK. */
#define RM_MAX_CHANNELS 6

static int dimensions(Py_ssize_t ne, Py_ssize_t n, Py_ssize_t c) {
    if (ne < 0 || n < 0 || c < 1 || c > RM_MAX_CHANNELS ||
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

/* Rows of the level sums on native threads. The caller holds no GIL; each
 * range writes its own rows, so the result is that of one sweep. Threads run
 * below normal priority (as the verification threads in parallel.py): the
 * machine is the user's. A thread that cannot be created runs its rows here. */
typedef void (*rm_range_fn)(void *,Py_ssize_t,Py_ssize_t);
typedef struct {rm_range_fn fn;void *context;Py_ssize_t lo,hi;} rm_part;
#define RM_MAX_THREADS 16
#ifdef _WIN32
#include <windows.h>
static DWORD WINAPI rm_part_run(LPVOID p) {rm_part *q=(rm_part *)p;q->fn(q->context,q->lo,q->hi);return 0;}
#else
#include <pthread.h>
static void *rm_part_run(void *p) {rm_part *q=(rm_part *)p;q->fn(q->context,q->lo,q->hi);return NULL;}
#endif
static void rm_parallel(Py_ssize_t ne,int threads,rm_range_fn fn,void *context) {
    if (threads>RM_MAX_THREADS) threads=RM_MAX_THREADS;
    if (threads<2 || ne<2*RM_BLOCK) {fn(context,0,ne);return;}
    Py_ssize_t chunk=((ne+threads-1)/threads+RM_BLOCK-1)/RM_BLOCK*RM_BLOCK;
    rm_part parts[RM_MAX_THREADS];int started[RM_MAX_THREADS]={0};int m=0;
#ifdef _WIN32
    HANDLE handles[RM_MAX_THREADS];
#else
    pthread_t handles[RM_MAX_THREADS];
#endif
    for (Py_ssize_t lo=chunk;lo<ne && m<RM_MAX_THREADS;lo+=chunk,m++) {
        parts[m].fn=fn;parts[m].context=context;parts[m].lo=lo;parts[m].hi=lo+chunk<ne?lo+chunk:ne;
#ifdef _WIN32
        handles[m]=CreateThread(NULL,0,rm_part_run,&parts[m],CREATE_SUSPENDED,NULL);
        if (handles[m]!=NULL) {
            SetThreadPriority(handles[m],THREAD_PRIORITY_BELOW_NORMAL);
            ResumeThread(handles[m]);started[m]=1;
        }
#else
        started[m]=pthread_create(&handles[m],NULL,rm_part_run,&parts[m])==0;
#endif
    }
    fn(context,0,chunk<ne?chunk:ne);
    for (int i=0;i<m;i++) {
        if (!started[i]) {fn(context,parts[i].lo,parts[i].hi);continue;}
#ifdef _WIN32
        WaitForSingleObject(handles[i],INFINITE);CloseHandle(handles[i]);
#else
        pthread_join(handles[i],NULL);
#endif
    }
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
#define RM_VECTOR 1
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
#undef RM_VECTOR
#endif
static int use_avx2=0;

static int envelope_bounded(Py_ssize_t ne,Py_ssize_t n,Py_ssize_t c,
        const double *gamma,const double *a,const double *f) {
    /* |Re(1/d)|+|Im(1/d)| <= 3/gamma. Outward rounding and
     * contraction margin certify the original per-energy envelope guard.
     * Dispatch to separate kernels so an unavailable certificate does not
     * add branches inside the existing generic inner loops.
     */
    double max_f2[RM_MAX_CHANNELS]={0},bound=0.;
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
    int threads=1;
    if (!PyArg_ParseTuple(args,"nnnOOOOOOO|i",&ne,&n,&c,&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6],&threads)) return NULL;
    if (!dimensions(ne,n,c)) return NULL;
    Py_ssize_t sizes[7]={ne,n,n,n*c,ne*c,2*ne*c*c,ne};
    for (int i=0;i<7;i++) if (!buffer(o[i],&b[i],sizes[i],i==6?"B":"d",i>=5)) {release(b);return NULL;}
    unsigned char *flags=PyMem_Malloc(n>0?(size_t)n:1);
    if (flags==NULL) {release(b);return PyErr_NoMemory();}
    Py_BEGIN_ALLOW_THREADS
    int certified=envelope_bounded(ne,n,c,b[2].buf,b[3].buf,b[4].buf);
#ifdef RM_HAS_AVX2
    if (use_avx2) {
        if (certified) matrix_sum_certified_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf,flags,threads);
        else matrix_sum_avx2(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf,flags,threads);
    }
    else
#endif
    if (certified) matrix_sum_certified(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf,flags,threads);
    else matrix_sum(ne,n,c,b[0].buf,b[1].buf,b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf,flags,threads);
    Py_END_ALLOW_THREADS
    PyMem_Free(flags);release(b);Py_RETURN_NONE;
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

typedef struct {double r,i;} rm_complex;
static rm_complex rm_sub(rm_complex a,rm_complex b) {
    rm_complex z={a.r-b.r,a.i-b.i};return z;
}
static rm_complex rm_mul(rm_complex a,rm_complex b) {
    rm_complex z={a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r};return z;
}
static rm_complex rm_div(rm_complex a,rm_complex b) {
    /* Scaled complex division, without squaring the pivot magnitude. */
    double ratio,den;rm_complex z;
    if (fabs(b.r)>=fabs(b.i)) {
        ratio=b.i/b.r;den=b.r+b.i*ratio;
        z.r=(a.r+a.i*ratio)/den;z.i=(a.i-a.r*ratio)/den;
    } else {
        ratio=b.r/b.i;den=b.i+b.r*ratio;
        z.r=(a.r*ratio+a.i)/den;z.i=(a.i*ratio-a.r)/den;
    }
    return z;
}

/* One c <= 3 channel system m y = e_entrance (m: 2*c*c doubles, row-major
 * complex) by Gaussian elimination with partial pivoting. No explicit
 * inverse or determinant formula; the callers check the residual,
 * passivity and flux of the candidate solution. 0 when it cannot. */
static int small_solve(Py_ssize_t c,Py_ssize_t entrance,const double *m,rm_complex *solution) {
    rm_complex a[9],rhs[3]={{0}};
    rhs[entrance].r=1.;
    for (Py_ssize_t j=0;j<c*c;j++) {
        a[j].r=m[2*j];a[j].i=m[2*j+1];
        if (!isfinite(a[j].r) || !isfinite(a[j].i) ||
                fabs(a[j].r)>1e8 || fabs(a[j].i)>1e8) return 0;
    }
    for (Py_ssize_t k=0;k<c;k++) {
        Py_ssize_t pivot=k;double largest=0.;
        for (Py_ssize_t j=k;j<c;j++) {
            double magnitude=hypot(a[j*c+k].r,a[j*c+k].i);
            if (magnitude>largest) {largest=magnitude;pivot=j;}
        }
        if (!(largest>=DBL_MIN && isfinite(largest))) return 0;
        if (pivot!=k) {
            for (Py_ssize_t j=k;j<c;j++) {
                rm_complex saved=a[k*c+j];a[k*c+j]=a[pivot*c+j];a[pivot*c+j]=saved;
            }
            rm_complex saved=rhs[k];rhs[k]=rhs[pivot];rhs[pivot]=saved;
        }
        for (Py_ssize_t j=k+1;j<c;j++) {
            rm_complex factor=rm_div(a[j*c+k],a[k*c+k]);
            for (Py_ssize_t l=k+1;l<c;l++) a[j*c+l]=rm_sub(a[j*c+l],rm_mul(factor,a[k*c+l]));
            rhs[j]=rm_sub(rhs[j],rm_mul(factor,rhs[k]));
        }
    }
    int ok=1;
    for (Py_ssize_t j=c;j-->0;) {
        rm_complex value=rhs[j];
        for (Py_ssize_t k=j+1;k<c;k++) value=rm_sub(value,rm_mul(a[j*c+k],solution[k]));
        solution[j]=rm_div(value,a[j*c+j]);
        if (!isfinite(solution[j].r) || !isfinite(solution[j].i)) ok=0;
    }
    return ok;
}

static PyObject *channel_solve(PyObject *self,PyObject *args) {
    Py_ssize_t ne,c,entrance;PyObject *input,*output;Py_buffer b[7]={0};
    if (!PyArg_ParseTuple(args,"nnnOO",&ne,&c,&entrance,&input,&output)) return NULL;
    if (!dimensions(ne,0,c)) return NULL;
    if (entrance<0 || entrance>=c || c>3) {
        PyErr_SetString(PyExc_ValueError,"invalid native entrance channel");return NULL;
    }
    if (!buffer(input,&b[0],2*ne*c*c,"d",0) || !buffer(output,&b[1],2*ne*c,"d",1)) {
        release(b);return NULL;
    }
    int failed=0;const double *matrix=b[0].buf;double *out=b[1].buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row=0;row<ne;row++) {
        rm_complex solution[3];
        int ok=small_solve(c,entrance,matrix+2*row*c*c,solution);
        for (Py_ssize_t j=0;j<c && ok;j++) {out[2*(row*c+j)]=solution[j].r;out[2*(row*c+j)+1]=solution[j].i;}
        if (!ok) {failed=1;break;}
    }
    Py_END_ALLOW_THREADS
    release(b);return PyBool_FromLong(!failed);
}

/* The rows of _rm_acceleration.solve after the level sums, fused: the
 * channel system (I - iR) y = e_entrance, its residual, W = R y, and the
 * Gram capture 4 Re(y* Im(R) y) with its roundoff majorant, bound and
 * acceptance, for c <= 3. Every output is computed with the operations, in
 * the order, of the NumPy statements it replaces (complex reciprocal for
 * c = 1 as NumPy's, einsum sums in index order), so the published values do
 * not change. Returns (status, maximum): status 1 means some system needs
 * the NumPy path; 2 means a non-passive |y| > 1 + 1e-8. */
static PyObject *solve_rows(PyObject *self,PyObject *args) {
    Py_ssize_t nr,c,entrance,nlev;PyObject *o[9];Py_buffer b[9]={0};
    double rtol,ermin,ermax,halfgmax2,hgmin;int scalar_ok,radiative_any;
    if (!PyArg_ParseTuple(args,"nnnnOOOOOOOOOdddddii",&nr,&c,&entrance,&nlev,
            &o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6],&o[7],&o[8],
            &rtol,&ermin,&ermax,&halfgmax2,&hgmin,&scalar_ok,&radiative_any)) return NULL;
    if (!dimensions(nr,nlev,c) || c>3 || entrance<0 || entrance>=c) {
        PyErr_SetString(PyExc_ValueError,"invalid native solve dimensions");return NULL;
    }
    /* r, f, e, active | y, w, cap, accepted, bound */
    Py_ssize_t sizes[9]={2*nr*c*c,nr*c,nr,c,2*nr*c,2*nr*c,nr,nr,nr};
    const char *formats[9]={"d","d","d","B","d","d","d","B","d"};
    for (int i=0;i<9;i++) if (!buffer(o[i],&b[i],sizes[i],formats[i],i>=4)) {
        for (int k=0;k<9;k++) if (b[k].obj!=NULL) PyBuffer_Release(&b[k]);
        return NULL;
    }
    const double *r=b[0].buf,*f=b[1].buf,*e=b[2].buf;const unsigned char *active=b[3].buf;
    double *y=b[4].buf,*w=b[5].buf,*cap=b[6].buf,*bound=b[8].buf;unsigned char *accepted=b[7].buf;
    int status=0;double maximum=0.;
    const double eps=DBL_EPSILON,tiny=DBL_MIN;
    double scale=8*((double)nlev+32*(double)c+64)*eps;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row=0;row<nr && !status;row++) {
        const double *rr=r+2*row*c*c;double m[18];rm_complex s[3];
        /* matrix = eye(c) - 1j*rr, as NumPy forms it */
        for (Py_ssize_t j=0;j<c;j++) for (Py_ssize_t l=0;l<c;l++) {
            double re=rr[2*(j*c+l)],im=rr[2*(j*c+l)+1];
            double pr=0.*re-1.*im,pi=0.*im+1.*re;
            m[2*(j*c+l)]=(j==l?1.:0.)-pr;m[2*(j*c+l)+1]=0.-pi;
        }
        if (c==1) {
            /* NumPy's 1/z (Smith's method with a reciprocal scale) */
            double zr=m[0],zi=m[1];
            if (fabs(zr)>=fabs(zi)) {
                if (zr==0. && zi==0.) {status=1;break;}
                double rat=zi/zr,scl=1.0/(zr+zi*rat);
                s[0].r=(1.+0.*rat)*scl;s[0].i=(0.-1.*rat)*scl;
            } else {
                double rat=zr/zi,scl=1.0/(zi+zr*rat);
                s[0].r=(1.*rat+0.)*scl;s[0].i=(0.*rat-1.)*scl;
            }
        } else if (!small_solve(c,entrance,m,s)) {status=1;break;}
        double norm2=0.,mnorm2=0.;
        for (Py_ssize_t j=0;j<c;j++) norm2+=s[j].r*s[j].r+s[j].i*s[j].i;
        if (sqrt(norm2)>1+1e-8) {status=2;break;}
        for (Py_ssize_t j=0;j<2*c*c;j++) mnorm2+=m[j]*m[j];
        double denominator=sqrt(mnorm2)*sqrt(norm2)+1.;
        for (Py_ssize_t j=0;j<c;j++) {
            double ar=(j==entrance?-1.:0.),ai=0.;
            for (Py_ssize_t l=0;l<c;l++) {
                ar+=m[2*(j*c+l)]*s[l].r-m[2*(j*c+l)+1]*s[l].i;
                ai+=m[2*(j*c+l)]*s[l].i+m[2*(j*c+l)+1]*s[l].r;
            }
            double ratio=hypot(ar,ai)/denominator;
            if (!(ratio<=maximum)) maximum=ratio;
        }
        for (Py_ssize_t j=0;j<c;j++) {y[2*(row*c+j)]=s[j].r;y[2*(row*c+j)+1]=s[j].i;}
        /* w = rr[:,0,:]*y (c=1) or einsum('ecd,ed->ec') */
        for (Py_ssize_t j=0;j<c;j++) {
            double accr=0.,acci=0.;
            for (Py_ssize_t l=0;l<c;l++) {
                double re=rr[2*(j*c+l)],im=rr[2*(j*c+l)+1];
                double pr=re*s[l].r-im*s[l].i,pi=re*s[l].i+im*s[l].r;
                if (c==1) {accr=pr;acci=pi;} else {accr+=pr;acci+=pi;}
            }
            w[2*(row*c+j)]=accr;w[2*(row*c+j)+1]=acci;
        }
        /* gram_capture */
        double value,major,diag[3];
        for (Py_ssize_t j=0;j<c;j++) diag[j]=rr[2*(j*c+j)+1];
        if (c==1) {
            double tr=diag[0]*s[0].r-0.*s[0].i,ti=diag[0]*s[0].i+0.*s[0].r;
            value=4*(s[0].r*tr-(-s[0].i)*ti);
            double t=4*(hypot(s[0].r,s[0].i)*sqrt(fmax(diag[0],0.)));
            major=t*t;
        } else {
            double sum=0.,msum=0.;
            for (Py_ssize_t i=0;i<c;i++) {
                double tr=0.,ti=0.;
                for (Py_ssize_t j=0;j<c;j++) {
                    double bij=rr[2*(i*c+j)+1];
                    tr+=bij*s[j].r-0.*s[j].i;ti+=bij*s[j].i+0.*s[j].r;
                }
                sum+=s[i].r*tr-(-s[i].i)*ti;
                msum+=hypot(s[i].r,s[i].i)*sqrt(fmax(diag[i],0.));
            }
            value=4*sum;major=4*(msum*msum);
        }
        double bnd=scale<1?scale/(1-scale)*major:INFINITY;
        int normal=scalar_ok;
        for (Py_ssize_t j=0;j<c;j++) {
            if (!(diag[j]>=tiny || !active[j])) normal=0;
            double ff=f[row*c+j]*f[row*c+j];
            if (!(ff>=tiny || f[row*c+j]==0)) normal=0;
        }
        if (radiative_any) {
            double distance=fmax(fabs(e[row]-ermin),fabs(e[row]-ermax));
            double upper=(distance*distance+halfgmax2)*(1+8*eps);
            double lower=nextafter(hgmin/upper,0.);
            if (!(lower>=tiny)) normal=0;
        }
        int ok=normal && isfinite(value) && value>=tiny && major>=tiny && bnd<=rtol*value && bnd<=1e-10;
        if (rtol==0.) ok=0;
        if (!radiative_any) {ok=1;value=0.;}
        cap[row]=value;accepted[row]=(unsigned char)ok;bound[row]=bnd;
    }
    Py_END_ALLOW_THREADS
    for (int k=0;k<9;k++) if (b[k].obj!=NULL) PyBuffer_Release(&b[k]);
    return Py_BuildValue("(id)",status,maximum);
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


/* URR probability-table ladders (kika.processing.resonances.probability_tables).
 * For each resonance, the sample energies inside its reach are found by binary
 * search on the sorted sample grid and its SLBW Doppler-broadened elastic,
 * capture and fission contributions added. The Doppler line shapes use
 * Humlicek's W4 approximation of the Faddeeva function (relative error about
 * 1e-4, the accuracy class of PURR's own rational approximations). */
/* exp, sin and cos for the Faddeeva function below. The MinGW-w64 UCRT
 * libm takes ~35 ns per exp and ~30 per sin or cos, and the fourth region of
 * Humlicek's W4 needs all three in every call: they were most of a URR
 * ladder. kexp is _doppler_native.c's (Cody-Waite + degree-13 Taylor,
 * <= 0.7 ulp); ksincos reduces by pi/2 with fdlibm's two-term split and uses
 * fdlibm's __kernel_sin/__kernel_cos polynomials (<= 1 ulp on |r| <= pi/4).
 * Both are checked in tests/test_probability_tables.py; W4 itself is good to
 * ~1e-4, so they change nothing it states. */
static double kexp(double x) {
    if(!(x>-745.2)) return x!=x?x:0.;
    if(x>709.78) return INFINITY;
    double k=floor(x*1.44269504088896338700+.5);
    double r=(x-k*6.93147180369123816490e-01)-k*1.90821492927058770002e-10;
    double p=1.+r*(1.+r*(1./2+r*(1./6+r*(1./24+r*(1./120+r*(1./720+r*(1./5040
             +r*(1./40320+r*(1./362880+r*(1./3628800+r*(1./39916800+r*(1./479001600
             +r*(1./6227020800.)))))))))))));
    int n=(int)k;
    if(n>=-1021) {
        uint64_t bits=(uint64_t)(n+1023)<<52;double scale;memcpy(&scale,&bits,sizeof scale);
        return p*scale;
    }
    return ldexp(p,n);
}

static void ksincos(double a,double *sine,double *cosine) {
    if(!(fabs(a)<1e5)) {*sine=sin(a);*cosine=cos(a);return;}
    double k=floor(a*6.36619772367581382433e-01+.5);
    double r=(a-k*1.57079632673412561417e+00)-k*6.07710050650619224932e-11;
    double z=r*r;
    double sr=r+r*z*(-1.66666666666666324348e-01+z*(8.33333333332248946124e-03+z*(-1.98412698298579493134e-04
              +z*(2.75573137070700676789e-06+z*(-2.50507602534068634195e-08+z*1.58969099521155010221e-10)))));
    double cr=1.-.5*z+z*z*(4.16666666666666019037e-02+z*(-1.38888888888741095749e-03+z*(2.48015872894767294178e-05
              +z*(-2.75573143513906633035e-07+z*(2.08757232129817482790e-09+z*-1.13596475577881948265e-11)))));
    switch(((long long)k)&3) {
        case 0: *sine=sr;*cosine=cr;break;
        case 1: *sine=cr;*cosine=-sr;break;
        case 2: *sine=-sr;*cosine=-cr;break;
        default: *sine=-cr;*cosine=sr;break;
    }
}

static void humlicek(double x,double y,double *wr,double *wi) {
    double s=fabs(x)+y;rm_complex t={y,-x},w;
    if (s>=15.) {
        rm_complex u=rm_mul(t,t);u.r+=0.5;
        w=rm_div((rm_complex){t.r*0.5641896,t.i*0.5641896},u);
    } else if (s>=5.5) {
        rm_complex u=rm_mul(t,t);
        rm_complex num=rm_mul(t,(rm_complex){1.410474+u.r*0.5641896,u.i*0.5641896});
        rm_complex den=rm_mul(u,(rm_complex){3.+u.r,u.i});den.r+=0.75;
        w=rm_div(num,den);
    } else if (y>=0.195*fabs(x)-0.176) {
        rm_complex num={3.778987+t.r*0.5642236,t.i*0.5642236};
        num=rm_mul(t,num);num.r+=11.96482;num=rm_mul(t,num);num.r+=20.20933;
        num=rm_mul(t,num);num.r+=16.4955;
        rm_complex den={6.699398+t.r,t.i};
        den=rm_mul(t,den);den.r+=21.69274;den=rm_mul(t,den);den.r+=39.27121;
        den=rm_mul(t,den);den.r+=38.82363;den=rm_mul(t,den);den.r+=16.4955;
        w=rm_div(num,den);
    } else {
        rm_complex u=rm_mul(t,t);
        static const double a[7]={36183.31,3321.9905,1540.787,219.0313,35.76683,1.320522,0.56419};
        static const double b[8]={32066.6,24322.84,9022.228,2186.181,364.2191,61.57037,1.841439,1.};
        rm_complex num={a[6],0.},den={b[7],0.};
        for (int k=5;k>=0;k--) {num=rm_mul(u,num);num.r=a[k]-num.r;num.i=-num.i;}
        for (int k=6;k>=0;k--) {den=rm_mul(u,den);den.r=b[k]-den.r;den.i=-den.i;}
        double magnitude=kexp(u.r),sine,cosine;ksincos(u.i,&sine,&cosine);
        rm_complex e={magnitude*cosine,magnitude*sine};
        rm_complex q=rm_div(rm_mul(t,num),den);
        w=rm_sub(e,q);
    }
    *wr=w.r;*wi=w.i;
}

#if defined(_MSC_VER) && !defined(restrict)
#define restrict __restrict
#endif
/* The URR ladder kernel, baseline and AVX2 (see _urr_kernel.h). No result of
 * it depends on traps, so the compiler may if-convert and vectorize. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-trapping-math")
#endif
#define URR_NAME(x) urr_##x##_base
#define URR_TARGET
#include "_urr_kernel.h"
#undef URR_NAME
#undef URR_TARGET
#ifdef RM_HAS_AVX2
#define URR_NAME(x) urr_##x##_avx2
#define URR_TARGET __attribute__((target("avx2")))
#include "_urr_kernel.h"
#undef URR_NAME
#undef URR_TARGET
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

static PyObject *urr_ladder(PyObject *self,PyObject *args) {
    Py_ssize_t ns,nr;double c2,s2,delta,reachw,reachd;PyObject *o[10];Py_buffer b[10]={0};
    if (!PyArg_ParseTuple(args,"nnddddd" "OOOOOOOOOO",&ns,&nr,&c2,&s2,&delta,&reachw,&reachd,
                          &o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6],&o[7],&o[8],&o[9])) return NULL;
    if (ns<0 || nr<0 || !isfinite(c2) || !isfinite(s2) || !(delta>=0) || !isfinite(delta)) {
        PyErr_SetString(PyExc_ValueError,"invalid URR ladder arguments");return NULL;
    }
    Py_ssize_t sizes[10]={ns,nr,nr,nr,nr,nr,nr,ns,ns,ns};
    for (int i=0;i<10;i++) {
        if (!buffer(o[i],&b[i],sizes[i],"d",i>=7)) {
            for (int j=0;j<10;j++) if (b[j].obj!=NULL) PyBuffer_Release(&b[j]);
            return NULL;
        }
    }
    const double *es=b[0].buf,*er=b[1].buf,*gt=b[2].buf,*gn=b[3].buf,*gg=b[4].buf,*gf=b[5].buf,*s0=b[6].buf;
    double *el=b[7].buf,*ca=b[8].buf,*fi=b[9].buf;
    Py_BEGIN_ALLOW_THREADS
#ifdef RM_HAS_AVX2
    if (use_avx2) urr_ladder_avx2(ns,nr,c2,s2,delta,reachw,reachd,es,er,gt,gn,gg,gf,s0,el,ca,fi);
    else
#endif
    urr_ladder_base(ns,nr,c2,s2,delta,reachw,reachd,es,er,gt,gn,gg,gf,s0,el,ca,fi);
    Py_END_ALLOW_THREADS
    for (int j=0;j<10;j++) if (b[j].obj!=NULL) PyBuffer_Release(&b[j]);
    Py_RETURN_NONE;
}

static PyObject *special_probe(PyObject *self,PyObject *args) {
    double x,y,sine,cosine,wr,wi;
    if(!PyArg_ParseTuple(args,"dd",&x,&y)) return NULL;
    ksincos(x,&sine,&cosine);humlicek(x,y,&wr,&wi);
    return Py_BuildValue("ddddd",kexp(-fabs(x)),sine,cosine,wr,wi);
}

static PyMethodDef methods[]={
    {"special",special_probe,METH_VARARGS,"Private kexp(-|x|), sin x, cos x, W4(x, y) diagnostic."},
    {"channel_solve",channel_solve,METH_VARARGS,"Solve small channel systems by pivoted elimination; false requires NumPy fallback."},
    {"solve_rows",solve_rows,METH_VARARGS,"Channel solve, residual, W and Gram capture of RM rows (c <= 3); status 1 requires NumPy fallback."},
    {"breit_wigner",breit_wigner,METH_VARARGS,"Accumulate all constant-radius BW levels in source order."},
    {"matrices",matrices,METH_VARARGS,"Fill all-level collision sums and reference-fallback flags."},
    {"absorption",absorption,METH_VARARGS,"Fill positive radiative probabilities using compensated level sums."},
    {"urr_ladder",urr_ladder,METH_VARARGS,"Accumulate SLBW Doppler-broadened URR ladder contributions on sorted samples."},
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
