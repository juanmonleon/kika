#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* NumPy's power is the C runtime's pow: glibc's on Linux, the Universal CRT's
 * (ucrtbase.dll) on Windows. MinGW links its own pow, 1 ulp apart on some
 * arguments, so on Windows the UCRT's is looked up at import. */
typedef double (*pow_function)(double, double);
static pow_function crt_pow = pow;
/* NumPy's cbrt is the C runtime's too, looked up the same way. */
typedef double (*cbrt_function)(double);
static cbrt_function crt_cbrt = cbrt;
static cbrt_function crt_tanh = tanh;   /* NumPy's tanh, the C runtime's as well */

/* lindhard's ep^(1/6) and ep^(3/4) as heating._lindhardPowers forms them:
 * sqrt(cbrt(ep)) and s*sqrt(s), s = sqrt(ep). */
static inline void lindhard_powers(double ep, double *sixth, double *threeQuarters)
{
    double s = sqrt(ep);
    *sixth = sqrt(crt_cbrt(ep));
    *threeQuarters = s * sqrt(s);
}

/* The damage quadratures of kika.processing.heating, operation for operation:
 * each line keeps the NumPy expression's order of operations, np.interp's
 * branches and np.sum's pairwise summation, so the result is bit for bit the
 * reference path's. No fused operations (the build uses strict floating point). */

/* NumPy's pairwise summation (DOUBLE_pairwise_sum), from 0. */
static double pairwise_sum(const double *a, Py_ssize_t n)
{
    if (n < 8) {
        double res = 0.;
        for (Py_ssize_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        Py_ssize_t i;
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    Py_ssize_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

/* np.sum of a whole contiguous array: the reduction runs over buffers of
 * NPY_BUFSIZE (8192) elements, each summed pairwise, accumulated in order. */
static double numpy_total(const double *a, Py_ssize_t n)
{
    double res = 0.;
    for (Py_ssize_t k = 0; k < n; k += 8192)
        res += pairwise_sum(a + k, (n - k) < 8192 ? (n - k) : 8192);
    return res;
}

/* np.maximum(a, b): a >= b or a is NaN gives a. */
static inline double np_maximum(double a, double b) { return (a >= b || isnan(a)) ? a : b; }
static inline double np_minimum(double a, double b) { return (a <= b || isnan(a)) ? a : b; }

/* kika.processing.heating.lindhard on one recoil, el and fl from the caller. */
static inline double lindhard1(double E, double el, double fl, double ed)
{
    double ep = np_maximum(E, 0.);
    ep /= el;
    double out, term;
    lindhard_powers(ep, &out, &term);
    out *= 3.4008;
    term *= 0.40244;
    out += term;
    out += ep;
    out *= fl;
    out += 1.0;
    out = E / out;
    return (E >= ed) ? out : 0.;
}

/* The active displacement model (kika.processing.heating.DisplacementModel):
 * mode 0 is MT444's damage energy, 1 NRT, 2 arc-dpa; Ed, b, c its constants. */
typedef struct { int mode; double Ed, b, c, high, coefficient; } displacement_model;

/* The constants of a model, once per call: (1 - c)/(2Ed/0.8)^b is the same value
 * NumPy forms at every recoil, so hoisting it changes no bit. */
static inline void displacement_prepare(displacement_model *m)
{
    m->high = 2. * m->Ed / 0.8;
    m->coefficient = (m->mode == 2) ? (1. - m->c) / crt_pow(m->high, m->b) : 0.;
}

/* DisplacementModel.count on one damage energy, in NumPy's order of operations. */
static inline double displacements1(double T, const displacement_model *m)
{
    double n = (0.8 * T) / (2. * m->Ed);
    if (m->mode == 2) n = n * (m->coefficient * crt_pow(T, m->b) + m->c);
    return (T < m->Ed) ? 0. : ((T < m->high) ? 1. : n);
}

/* lindhard1, or the displacements of its damage energy (no threshold on E then). */
static inline double damage1(double E, double el, double fl, double ed, const displacement_model *m)
{
    if (m->mode == 0) return lindhard1(E, el, fl, ed);
    double ep = np_maximum(E, 0.);
    ep /= el;
    double out, term;
    lindhard_powers(ep, &out, &term);
    out *= 3.4008;
    term *= 0.40244;
    out += term;
    out += ep;
    out *= fl;
    out += 1.0;
    out = E / out;
    return displacements1(out, m);
}

/* np.interp(q, x, p) for one point (finite tables, default left/right). */
static inline double np_interp1(double q, const double *x, const double *p, Py_ssize_t n)
{
    if (isnan(q)) return q;
    if (q < x[0]) return p[0];
    if (q > x[n - 1]) return p[n - 1];
    Py_ssize_t lo = 0, hi = n;          /* largest j with x[j] <= q */
    while (hi - lo > 1) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= q) lo = mid; else hi = mid;
    }
    Py_ssize_t j = lo;
    if (j == n - 1) return p[j];
    if (x[j] == q) return p[j];
    double slope = (p[j + 1] - p[j]) / (x[j + 1] - x[j]);
    double res = slope * (q - x[j]) + p[j];
    if (isnan(res)) {
        res = slope * (q - x[j + 1]) + p[j + 1];
        if (isnan(res) && p[j] == p[j + 1]) res = p[j];
    }
    return res;
}

/* np.searchsorted(x, q, side='right'): elements <= q. */
static inline Py_ssize_t search_right(double q, const double *x, Py_ssize_t n)
{
    Py_ssize_t lo = 0, hi = n;
    while (lo < hi) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= q) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static int get_vector(PyObject *object, Py_buffer *view, const char *what)
{
    if (PyObject_GetBuffer(object, view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return -1;
    if (view->itemsize != sizeof(double) || !view->format ||
        (strcmp(view->format, "d") && strcmp(view->format, "=d") && strcmp(view->format, "@d")) ||
        view->len % sizeof(double) || view->ndim > 1 || (uintptr_t)view->buf % sizeof(double)) {
        PyErr_Format(PyExc_ValueError, "native heating: %s must be a contiguous float64 vector", what);
        return -1;
    }
    return 0;
}

/* One chunk of _tabulatedEmissionDamage's at(E): the sum NumPy forms as
 *   np.sum(w*half[:,None]*2*v*density*oldwidth/width*angular)
 * over the panels whose sqrt-edges are sqrt(cuts[k]), sqrt(cuts[k+1]). */
static PyObject *emission_chunk(PyObject *self, PyObject *args)
{
    PyObject *objects[7];
    Py_buffer b[7] = {{0}};
    static const char *names[7] = {"cuts", "g", "w", "mu", "wm", "x", "p"};
    double low, width, oldwidth, incident, denominator, edDenominator, el, fl, ed;
    int hist, lab, noRecoil;
    displacement_model model = {0, 0., 0., 1., 0., 0.};
    double *terms = NULL, *row = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOdddddddddppp|iddd", &objects[0], &objects[1], &objects[2],
                          &objects[3], &objects[4], &objects[5], &objects[6], &low, &width, &oldwidth,
                          &incident, &denominator, &edDenominator, &el, &fl, &ed, &hist, &lab, &noRecoil,
                          &model.mode, &model.Ed, &model.b, &model.c))
        return NULL;
    for (int k = 0; k < 7; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *cuts = b[0].buf, *g = b[1].buf, *w = b[2].buf, *mu = b[3].buf, *wm = b[4].buf;
    const double *x = b[5].buf, *p = b[6].buf;
    Py_ssize_t nk = b[0].len / sizeof(double), n = b[1].len / sizeof(double);
    Py_ssize_t m = b[3].len / sizeof(double), nx = b[5].len / sizeof(double);
    if (nk < 2 || n < 1 || b[2].len != b[1].len || b[4].len != b[3].len || nx < 2 || b[6].len != b[5].len
        || (!lab && m < 1)) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent chunk buffers");
        goto done;
    }
    if (model.mode < 0 || model.mode > 2 || (model.mode && !(model.Ed > 0))) {
        PyErr_SetString(PyExc_ValueError, "native heating: invalid displacement model");
        goto done;
    }
    displacement_prepare(&model);
    Py_ssize_t nc = nk - 1;
    terms = PyMem_Malloc(sizeof(double) * (size_t)(nc * n));
    row = PyMem_Malloc(sizeof(double) * (size_t)(m > 0 ? m : 1));
    if (!terms || !row) { PyErr_NoMemory(); goto done; }
    double total;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nc; ++i) {
        double aa = sqrt(cuts[i]), bb = sqrt(cuts[i + 1]);
        double half = (bb - aa) / 2;
        double mid = (aa + bb) / 2;
        for (Py_ssize_t j = 0; j < n; ++j) {
            double v = mid + half * g[j];
            double ep = v * v;
            double source = x[0] + ((ep - low) * oldwidth) / width;
            double density;
            if (hist) {
                Py_ssize_t pos = search_right(source, x, nx) - 1;
                if (pos < 0) pos = 0;
                if (pos > nx - 2) pos = nx - 2;
                density = p[pos];
            } else {
                density = np_interp1(source, x, p, nx);
            }
            double angular;
            if (lab) {
                angular = noRecoil ? 0. : damage1(ep, el, fl, ed, &model);
            } else {
                double root = sqrt(incident * ep);
                double last = (root > 0) ? ((incident + ep) - edDenominator) / (2 * root) : 1.;
                last = np_minimum(np_maximum(last, -1.), 1.);
                for (Py_ssize_t k = 0; k < m; ++k) {
                    double angle = (last - 1) / 2 + ((last + 1) / 2) * mu[k];
                    double recoil = np_maximum(0., ((incident + ep) - (2 * root) * angle) / denominator);
                    row[k] = wm[k] * (noRecoil ? 0. : damage1(recoil, el, fl, ed, &model));
                }
                angular = (pairwise_sum(row, m) * (last + 1)) / 4;
            }
            terms[i * n + j] = (((((w[j] * half) * 2) * v) * density) * oldwidth) / width * angular;
        }
    }
    total = numpy_total(terms, nc * n);
    Py_END_ALLOW_THREADS
    result = PyFloat_FromDouble(total);
 done:
    PyMem_Free(terms);
    PyMem_Free(row);
    for (int k = 0; k < 7; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* One angular quadrature of _twoBodyDamage's at(E), before dividing by the
 * table's mass: np.sum(half[:,None]*w*density*lindhard(np.maximum(offset-slope*mu,0.)))
 * with mu = mid + half*u on the panels between consecutive cuts. */
static PyObject *twobody_chunk(PyObject *self, PyObject *args)
{
    PyObject *objects[5];
    Py_buffer b[5] = {{0}};
    static const char *names[5] = {"cuts", "u", "w", "x", "p"};
    double offset, slope, el, fl, ed;
    int hist;
    displacement_model model = {0, 0., 0., 1., 0., 0.};
    double *terms = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOdddddp|iddd", &objects[0], &objects[1], &objects[2], &objects[3],
                          &objects[4], &offset, &slope, &el, &fl, &ed, &hist,
                          &model.mode, &model.Ed, &model.b, &model.c))
        return NULL;
    for (int k = 0; k < 5; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *cuts = b[0].buf, *u = b[1].buf, *w = b[2].buf, *x = b[3].buf, *p = b[4].buf;
    Py_ssize_t nk = b[0].len / sizeof(double), n = b[1].len / sizeof(double), nx = b[3].len / sizeof(double);
    if (nk < 2 || n < 1 || b[2].len != b[1].len || nx < 2 || b[4].len != b[3].len) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent two-body buffers");
        goto done;
    }
    if (model.mode < 0 || model.mode > 2 || (model.mode && !(model.Ed > 0))) {
        PyErr_SetString(PyExc_ValueError, "native heating: invalid displacement model");
        goto done;
    }
    displacement_prepare(&model);
    Py_ssize_t nc = nk - 1;
    terms = PyMem_Malloc(sizeof(double) * (size_t)(nc * n));
    if (!terms) { PyErr_NoMemory(); goto done; }
    double total;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nc; ++i) {
        double half = (cuts[i + 1] - cuts[i]) / 2;
        double mid = (cuts[i] + cuts[i + 1]) / 2;
        for (Py_ssize_t j = 0; j < n; ++j) {
            double mu = mid + half * u[j];
            double density;
            if (hist) {
                Py_ssize_t pos = search_right(mu, x, nx) - 1;
                if (pos < 0) pos = 0;
                if (pos > nx - 1) pos = nx - 1;
                density = p[pos];
            } else {
                density = np_interp1(mu, x, p, nx);
            }
            double recoil = np_maximum(offset - slope * mu, 0.);
            terms[i * n + j] = ((half * w[j]) * density) * damage1(recoil, el, fl, ed, &model);
        }
    }
    total = numpy_total(terms, nc * n);
    Py_END_ALLOW_THREADS
    result = PyFloat_FromDouble(total);
 done:
    PyMem_Free(terms);
    for (int k = 0; k < 5; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}


/* The joint damage of one recoil: MT444's damage energy (lindhard1, with its ed
 * cut on the recoil energy) and the displacements of up to two models, from one
 * evaluation of the Lindhard partition. out[0] is bit for bit lindhard1. */
static inline void damage3(double E, double el, double fl, double ed, const displacement_model *m1,
                           const displacement_model *m2, double out[3])
{
    double ep = np_maximum(E, 0.);
    ep /= el;
    double T, term;
    lindhard_powers(ep, &T, &term);
    T *= 3.4008;
    term *= 0.40244;
    T += term;
    T += ep;
    T *= fl;
    T += 1.0;
    T = E / T;
    out[0] = (E >= ed) ? T : 0.;
    out[1] = m1->mode ? displacements1(T, m1) : 0.;
    out[2] = m2->mode ? displacements1(T, m2) : 0.;
}

static int parse_models(displacement_model *m1, displacement_model *m2)
{
    for (int k = 0; k < 2; ++k) {
        displacement_model *m = k ? m2 : m1;
        if (m->mode < 0 || m->mode > 2 || (m->mode && !(m->Ed > 0))) {
            PyErr_SetString(PyExc_ValueError, "native heating: invalid displacement model");
            return -1;
        }
        displacement_prepare(m);
    }
    return 0;
}

/* emission_chunk for MT444 and two models at once: a 3-tuple of its sums.
 * edD[c] = denominator x the recoil threshold of component c: the angular range
 * runs from -1 to the cosine where the recoil meets the lowest of them, split at
 * every other one, so that no Gauss panel holds a component's step; ed is
 * MT444's cut. */
static PyObject *emission_chunk_joint(PyObject *self, PyObject *args)
{
    PyObject *objects[7];
    Py_buffer b[7] = {{0}};
    static const char *names[7] = {"cuts", "g", "w", "mu", "wm", "x", "p"};
    double low, width, oldwidth, incident, denominator, edD[3], el, fl, ed;
    int hist, lab, noRecoil;
    displacement_model m1 = {0, 0., 0., 1., 0., 0.}, m2 = {0, 0., 0., 1., 0., 0.};
    double *terms = NULL, *row = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOdddddddddddpppidddiddd", &objects[0], &objects[1], &objects[2],
                          &objects[3], &objects[4], &objects[5], &objects[6], &low, &width, &oldwidth,
                          &incident, &denominator, &edD[0], &edD[1], &edD[2], &el, &fl, &ed, &hist, &lab, &noRecoil,
                          &m1.mode, &m1.Ed, &m1.b, &m1.c, &m2.mode, &m2.Ed, &m2.b, &m2.c))
        return NULL;
    for (int k = 0; k < 7; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *cuts = b[0].buf, *g = b[1].buf, *w = b[2].buf, *mu = b[3].buf, *wm = b[4].buf;
    const double *x = b[5].buf, *p = b[6].buf;
    Py_ssize_t nk = b[0].len / sizeof(double), n = b[1].len / sizeof(double);
    Py_ssize_t m = b[3].len / sizeof(double), nx = b[5].len / sizeof(double);
    if (nk < 2 || n < 1 || b[2].len != b[1].len || b[4].len != b[3].len || nx < 2 || b[6].len != b[5].len
        || (!lab && m < 1)) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent chunk buffers");
        goto done;
    }
    if (parse_models(&m1, &m2) < 0) goto done;
    Py_ssize_t nc = nk - 1;
    terms = PyMem_Malloc(sizeof(double) * (size_t)(3 * nc * n));
    row = PyMem_Malloc(sizeof(double) * (size_t)(3 * (m > 0 ? m : 1)));
    if (!terms || !row) { PyErr_NoMemory(); goto done; }
    double total[3];
    /* The thresholds in decreasing order (the cosines then increase). */
    double thr[3] = {edD[0], edD[1], edD[2]};
    for (int s = 0; s < 2; ++s)
        for (int q = 0; q < 2 - s; ++q)
            if (thr[q] < thr[q + 1]) { double tmp = thr[q]; thr[q] = thr[q + 1]; thr[q + 1] = tmp; }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nc; ++i) {
        double aa = sqrt(cuts[i]), bb = sqrt(cuts[i + 1]);
        double half = (bb - aa) / 2;
        double mid = (aa + bb) / 2;
        for (Py_ssize_t j = 0; j < n; ++j) {
            double v = mid + half * g[j];
            double ep = v * v;
            double source = x[0] + ((ep - low) * oldwidth) / width;
            double density;
            if (hist) {
                Py_ssize_t pos = search_right(source, x, nx) - 1;
                if (pos < 0) pos = 0;
                if (pos > nx - 2) pos = nx - 2;
                density = p[pos];
            } else {
                density = np_interp1(source, x, p, nx);
            }
            double angular[3];
            if (lab) {
                if (noRecoil) angular[0] = angular[1] = angular[2] = 0.;
                else damage3(ep, el, fl, ed, &m1, &m2, angular);
            } else {
                double root = sqrt(incident * ep);
                angular[0] = angular[1] = angular[2] = 0.;
                double start = -1.;
                for (int s = 0; s < 3; ++s) {
                    if (s && thr[s] == thr[s - 1]) continue;
                    double stop = (root > 0) ? ((incident + ep) - thr[s]) / (2 * root) : 1.;
                    stop = np_minimum(np_maximum(stop, -1.), 1.);
                    if (!(stop > start)) continue;
                    for (Py_ssize_t k = 0; k < m; ++k) {
                        double angle = (start + stop) / 2 + ((stop - start) / 2) * mu[k];
                        double recoil = np_maximum(0., ((incident + ep) - (2 * root) * angle) / denominator);
                        double d[3] = {0., 0., 0.};
                        if (!noRecoil) damage3(recoil, el, fl, ed, &m1, &m2, d);
                        for (int c = 0; c < 3; ++c) row[c * m + k] = wm[k] * d[c];
                    }
                    for (int c = 0; c < 3; ++c) angular[c] += (pairwise_sum(row + c * m, m) * (stop - start)) / 4;
                    start = stop;
                }
            }
            double factor = (((((w[j] * half) * 2) * v) * density) * oldwidth) / width;
            for (int c = 0; c < 3; ++c) terms[c * nc * n + i * n + j] = factor * angular[c];
        }
    }
    for (int c = 0; c < 3; ++c) total[c] = numpy_total(terms + c * nc * n, nc * n);
    Py_END_ALLOW_THREADS
    result = Py_BuildValue("(ddd)", total[0], total[1], total[2]);
 done:
    PyMem_Free(terms);
    PyMem_Free(row);
    for (int k = 0; k < 7; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* twobody_chunk for MT444 and two models at once: a 3-tuple of its sums. */
static PyObject *twobody_chunk_joint(PyObject *self, PyObject *args)
{
    PyObject *objects[5];
    Py_buffer b[5] = {{0}};
    static const char *names[5] = {"cuts", "u", "w", "x", "p"};
    double offset, slope, el, fl, ed;
    int hist;
    displacement_model m1 = {0, 0., 0., 1., 0., 0.}, m2 = {0, 0., 0., 1., 0., 0.};
    double *terms = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOdddddpidddiddd", &objects[0], &objects[1], &objects[2], &objects[3],
                          &objects[4], &offset, &slope, &el, &fl, &ed, &hist,
                          &m1.mode, &m1.Ed, &m1.b, &m1.c, &m2.mode, &m2.Ed, &m2.b, &m2.c))
        return NULL;
    for (int k = 0; k < 5; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *cuts = b[0].buf, *u = b[1].buf, *w = b[2].buf, *x = b[3].buf, *p = b[4].buf;
    Py_ssize_t nk = b[0].len / sizeof(double), n = b[1].len / sizeof(double), nx = b[3].len / sizeof(double);
    if (nk < 2 || n < 1 || b[2].len != b[1].len || nx < 2 || b[4].len != b[3].len) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent two-body buffers");
        goto done;
    }
    if (parse_models(&m1, &m2) < 0) goto done;
    Py_ssize_t nc = nk - 1;
    terms = PyMem_Malloc(sizeof(double) * (size_t)(3 * nc * n));
    if (!terms) { PyErr_NoMemory(); goto done; }
    double total[3];
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nc; ++i) {
        double half = (cuts[i + 1] - cuts[i]) / 2;
        double mid = (cuts[i] + cuts[i + 1]) / 2;
        for (Py_ssize_t j = 0; j < n; ++j) {
            double mu = mid + half * u[j];
            double density;
            if (hist) {
                Py_ssize_t pos = search_right(mu, x, nx) - 1;
                if (pos < 0) pos = 0;
                if (pos > nx - 1) pos = nx - 1;
                density = p[pos];
            } else {
                density = np_interp1(mu, x, p, nx);
            }
            double recoil = np_maximum(offset - slope * mu, 0.);
            double d[3];
            damage3(recoil, el, fl, ed, &m1, &m2, d);
            double factor = (half * w[j]) * density;
            for (int c = 0; c < 3; ++c) terms[c * nc * n + i * n + j] = factor * d[c];
        }
    }
    for (int c = 0; c < 3; ++c) total[c] = numpy_total(terms + c * nc * n, nc * n);
    Py_END_ALLOW_THREADS
    result = Py_BuildValue("(ddd)", total[0], total[1], total[2]);
 done:
    PyMem_Free(terms);
    for (int k = 0; k < 5; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* The momenta of two independent particles, binned as _MomentumBins.moments gives them
 * (a: z, var z, p_perp, <p_perp^2>, w; b the same), added over the relative azimuth at the
 * nodes cos[k]: the points of _addMomenta, block by block of `rows` rows of a, in NumPy's
 * order (pair (i, j), then the azimuth). */
typedef struct {
    const double *az, *avz, *ap, *ap2, *aw, *bz, *bvz, *bp, *bp2, *bw, *cos;
    Py_ssize_t na, nb, nk;
} momenta_pair;

static int momenta_buffers(PyObject **objects, Py_buffer *b, momenta_pair *m)
{
    static const char *names[11] = {"az", "avz", "ap", "ap2", "aw", "bz", "bvz", "bp", "bp2", "bw", "cos"};
    for (int k = 0; k < 11; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) return -1;
    m->az = b[0].buf; m->avz = b[1].buf; m->ap = b[2].buf; m->ap2 = b[3].buf; m->aw = b[4].buf;
    m->bz = b[5].buf; m->bvz = b[6].buf; m->bp = b[7].buf; m->bp2 = b[8].buf; m->bw = b[9].buf;
    m->cos = b[10].buf;
    m->na = b[0].len / sizeof(double); m->nb = b[5].len / sizeof(double); m->nk = b[10].len / sizeof(double);
    for (int k = 1; k < 5; ++k)
        if (b[k].len != b[0].len || b[k + 5].len != b[5].len) {
            PyErr_SetString(PyExc_ValueError, "native heating: inconsistent momenta buffers");
            return -1;
        }
    if (m->na < 1 || m->nb < 1 || m->nk < 1) {
        PyErr_SetString(PyExc_ValueError, "native heating: empty momenta");
        return -1;
    }
    return 0;
}

/* _addMomenta(a, b, last=damage) of _independentRecoilDamage: the sum over every point of
 *   w * df(((P - z)**2 + var z + p_perp**2) / (2 ar))
 * each block's np.sum (pairwise) added in order, bit for bit the NumPy reference. */
static PyObject *momenta_damage(PyObject *self, PyObject *args)
{
    PyObject *objects[11];
    Py_buffer b[11] = {{0}};
    momenta_pair m;
    double P, denominator, el, fl, ed;
    Py_ssize_t rows;
    int noRecoil;
    displacement_model model = {0, 0., 0., 1., 0., 0.};
    double *terms = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOdddddnp|iddd", &objects[0], &objects[1], &objects[2],
                          &objects[3], &objects[4], &objects[5], &objects[6], &objects[7], &objects[8],
                          &objects[9], &objects[10], &P, &denominator, &el, &fl, &ed, &rows, &noRecoil,
                          &model.mode, &model.Ed, &model.b, &model.c))
        return NULL;
    if (momenta_buffers(objects, b, &m) < 0) goto done;
    if (rows < 1 || model.mode < 0 || model.mode > 2 || (model.mode && !(model.Ed > 0))) {
        PyErr_SetString(PyExc_ValueError, "native heating: invalid block or displacement model");
        goto done;
    }
    displacement_prepare(&model);
    Py_ssize_t block = rows < m.na ? rows : m.na;
    terms = PyMem_Malloc(sizeof(double) * (size_t)(block * m.nb * m.nk));
    if (!terms) { PyErr_NoMemory(); goto done; }
    double total = 0.;
    double nk = (double)m.nk;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < m.na; s += rows) {
        Py_ssize_t end = (s + rows < m.na) ? s + rows : m.na, t = 0;
        for (Py_ssize_t i = s; i < end; ++i)
            for (Py_ssize_t j = 0; j < m.nb; ++j) {
                double z = m.az[i] + m.bz[j];
                double vz = m.avz[i] + m.bvz[j];
                double w = (m.aw[i] * m.bw[j]) / nk;
                double ab = 2.0 * (m.ap[i] * m.bp[j]);
                double sq = m.ap2[i] + m.bp2[j];
                double d = P - z;
                for (Py_ssize_t k = 0; k < m.nk; ++k) {
                    double perp2 = np_maximum(sq + ab * m.cos[k], 0.);
                    double er = ((d * d + vz) + perp2) / denominator;
                    terms[t++] = w * (noRecoil ? 0. : damage1(er, el, fl, ed, &model));
                }
            }
        total += numpy_total(terms, t);
    }
    Py_END_ALLOW_THREADS
    result = PyFloat_FromDouble(total);
 done:
    PyMem_Free(terms);
    for (int k = 0; k < 11; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* _addMomenta(a, b) without `last`: the points binned into _MomentumBins, the five sums
 * (w, w z, w p_perp, w (z^2 + var z), w p_perp^2) per cell as np.bincount adds them (in the
 * points' order), block by block. Returns the (5, n*n) sums as bytes. */
static PyObject *momenta_bin(PyObject *self, PyObject *args)
{
    PyObject *objects[11];
    Py_buffer b[11] = {{0}};
    momenta_pair m;
    double zlo, zw, plo, pw;
    Py_ssize_t rows, n;
    double *sums = NULL, *block = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOddddnn", &objects[0], &objects[1], &objects[2],
                          &objects[3], &objects[4], &objects[5], &objects[6], &objects[7], &objects[8],
                          &objects[9], &objects[10], &zlo, &zw, &plo, &pw, &n, &rows))
        return NULL;
    if (momenta_buffers(objects, b, &m) < 0) goto done;
    if (rows < 1 || n < 1) {
        PyErr_SetString(PyExc_ValueError, "native heating: invalid bins");
        goto done;
    }
    Py_ssize_t cells = n * n;
    sums = PyMem_Calloc((size_t)(5 * cells), sizeof(double));
    block = PyMem_Calloc((size_t)(5 * cells), sizeof(double));
    if (!sums || !block) { PyErr_NoMemory(); goto done; }
    double nk = (double)m.nk, nn = (double)n;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < m.na; s += rows) {
        Py_ssize_t end = (s + rows < m.na) ? s + rows : m.na;
        memset(block, 0, sizeof(double) * (size_t)(5 * cells));
        for (Py_ssize_t i = s; i < end; ++i)
            for (Py_ssize_t j = 0; j < m.nb; ++j) {
                double z = m.az[i] + m.bz[j];
                double vz = m.avz[i] + m.bvz[j];
                double w = (m.aw[i] * m.bw[j]) / nk;
                double ab = 2.0 * (m.ap[i] * m.bp[j]);
                double sq = m.ap2[i] + m.bp2[j];
                for (Py_ssize_t k = 0; k < m.nk; ++k) {
                    double pp = sqrt(np_maximum(sq + ab * m.cos[k], 0.));
                    double fi = ((z - zlo) / zw) * nn, fk = ((pp - plo) / pw) * nn;
                    int64_t ci = (int64_t)fi, ck = (int64_t)fk;   /* astype(int64): toward zero */
                    ci = ci < 0 ? 0 : (ci > n - 1 ? n - 1 : ci);
                    ck = ck < 0 ? 0 : (ck > n - 1 ? n - 1 : ck);
                    Py_ssize_t cell = (Py_ssize_t)(ci * n + ck);
                    block[cell] += w;
                    block[cells + cell] += w * z;
                    block[2 * cells + cell] += w * pp;
                    block[3 * cells + cell] += w * (z * z + vz);
                    block[4 * cells + cell] += w * pp * pp;
                }
            }
        for (Py_ssize_t c = 0; c < 5 * cells; ++c) sums[c] += block[c];
    }
    Py_END_ALLOW_THREADS
    result = PyBytes_FromStringAndSize((const char *)sums, (Py_ssize_t)(sizeof(double) * 5 * cells));
 done:
    PyMem_Free(sums);
    PyMem_Free(block);
    for (int k = 0; k < 11; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* _recoilDamage's rows: for each outgoing energy e[i], the sum over the cosines mu[k] of
 *   weight[i, k] * df(max(E_R(E, e[i], mu[k]), 0))
 * with, in the centre of mass, E_R = a + (mp e)/M - (2 sqrt(((a mp) e)/M)) mu and, in the lab,
 * E_R = ((E - (2 sqrt((E mp) e)) mu) + mp e)/M; `a`, `a mp` and `E mp` come formed by the
 * caller, as NumPy forms them from scalars. A row is np.sum's pairwise sum over the cosines;
 * `weights` holds one row shared by every e, or one row per e. */
static PyObject *recoil_rows(PyObject *self, PyObject *args)
{
    PyObject *objects[3];
    Py_buffer b[3] = {{0}};
    static const char *names[3] = {"e", "mu", "weights"};
    double E, a, amp, Emp, mp, M, el, fl, ed;
    int cm, noRecoil;
    displacement_model model = {0, 0., 0., 1., 0., 0.};
    double *row = NULL, *out = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOdddddddddpp|iddd", &objects[0], &objects[1], &objects[2], &E, &a, &amp,
                          &Emp, &mp, &M, &el, &fl, &ed, &cm, &noRecoil,
                          &model.mode, &model.Ed, &model.b, &model.c))
        return NULL;
    for (int k = 0; k < 3; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *e = b[0].buf, *mu = b[1].buf, *weights = b[2].buf;
    Py_ssize_t n = b[0].len / sizeof(double), m = b[1].len / sizeof(double), nw = b[2].len / sizeof(double);
    /* one row per e, or (when there are as many as one row) shared: the same values */
    int perRow = (nw == n * m && n > 1);
    if (m < 1 || !(nw == m || nw == n * m)) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent recoil buffers");
        goto done;
    }
    if (model.mode < 0 || model.mode > 2 || (model.mode && !(model.Ed > 0))) {
        PyErr_SetString(PyExc_ValueError, "native heating: invalid displacement model");
        goto done;
    }
    displacement_prepare(&model);
    row = PyMem_Malloc(sizeof(double) * (size_t)m);
    out = PyMem_Malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    if (!row || !out) { PyErr_NoMemory(); goto done; }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        double ei = e[i], first, root;
        if (cm) {
            first = a + (mp * ei) / M;
            root = 2.0 * sqrt((amp * ei) / M);
        } else {
            first = 0.;
            root = 2.0 * sqrt(Emp * ei);
        }
        const double *w = perRow ? weights + i * m : weights;
        for (Py_ssize_t k = 0; k < m; ++k) {
            double er = cm ? first - root * mu[k] : ((E - root * mu[k]) + mp * ei) / M;
            er = np_maximum(er, 0.);
            row[k] = (noRecoil ? 0. : damage1(er, el, fl, ed, &model)) * w[k];
        }
        out[i] = pairwise_sum(row, m);
    }
    Py_END_ALLOW_THREADS
    result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)(sizeof(double) * n));
 done:
    PyMem_Free(row);
    PyMem_Free(out);
    for (int k = 0; k < 3; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* _MomentumBins.add of points (pz, pp, w) with no variance: the five sums per cell
 * (w, w pz, w pp, w pz^2, w pp^2) as np.bincount adds them, in the points' order, onto
 * zeros. Returns the (5, n*n) sums as bytes. */
static PyObject *points_bin(PyObject *self, PyObject *args)
{
    PyObject *objects[3];
    Py_buffer b[3] = {{0}};
    static const char *names[3] = {"pz", "pp", "w"};
    double zlo, zw, plo, pw;
    Py_ssize_t n;
    double *sums = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOddddn", &objects[0], &objects[1], &objects[2], &zlo, &zw, &plo, &pw, &n))
        return NULL;
    for (int k = 0; k < 3; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    Py_ssize_t np_ = b[0].len / sizeof(double);
    if (b[1].len != b[0].len || b[2].len != b[0].len || n < 1) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent points");
        goto done;
    }
    const double *pz = b[0].buf, *pp = b[1].buf, *w = b[2].buf;
    Py_ssize_t cells = n * n;
    sums = PyMem_Calloc((size_t)(5 * cells), sizeof(double));
    if (!sums) { PyErr_NoMemory(); goto done; }
    double nn = (double)n;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t t = 0; t < np_; ++t) {
        double z = pz[t], q = pp[t], wt = w[t];
        int64_t ci = (int64_t)(((z - zlo) / zw) * nn), ck = (int64_t)(((q - plo) / pw) * nn);
        ci = ci < 0 ? 0 : (ci > n - 1 ? n - 1 : ci);
        ck = ck < 0 ? 0 : (ck > n - 1 ? n - 1 : ck);
        Py_ssize_t cell = (Py_ssize_t)(ci * n + ck);
        sums[cell] += wt;
        sums[cells + cell] += wt * z;
        sums[2 * cells + cell] += wt * q;
        sums[3 * cells + cell] += wt * (z * z + 0.0);
        sums[4 * cells + cell] += wt * q * q;
    }
    Py_END_ALLOW_THREADS
    result = PyBytes_FromStringAndSize((const char *)sums, (Py_ssize_t)(sizeof(double) * 5 * cells));
 done:
    PyMem_Free(sums);
    for (int k = 0; k < 3; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

/* One quadrature of _unitBaseCMMean's at(E): with the table's support [a, a+width] mapped
 * to [qa, qa+qw] and r the Gauss nodes in sqrt(E') on each panel,
 *   np.sum((right-left)/2*ww*2*r*r*width/qw*density*angle)/norm
 * with density, and angle (mu, or Kalbach r * Langevin(a)), read lin-lin (or held) at the
 * source energy of each node. Bit for bit heating._unitBaseIntegralNumpy. */
static PyObject *unit_base_integral(PyObject *self, PyObject *args)
{
    PyObject *objects[6];
    Py_buffer b[6] = {{0}};
    static const char *names[6] = {"x", "p", "angle", "slope", "nodes", "weights"};
    double a, width, qa, qw;
    int flat, kalbach;
    double *terms = NULL, *dnorm = NULL;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOddddpp", &objects[0], &objects[1], &objects[2], &objects[3],
                          &objects[4], &objects[5], &a, &width, &qa, &qw, &flat, &kalbach))
        return NULL;
    for (int k = 0; k < 6; ++k)
        if (get_vector(objects[k], &b[k], names[k]) < 0) goto done;
    const double *x = b[0].buf, *p = b[1].buf, *ang = b[2].buf, *sl = b[3].buf;
    const double *nodes = b[4].buf, *ww = b[5].buf;
    Py_ssize_t n = b[0].len / sizeof(double), m = b[4].len / sizeof(double);
    if (n < 2 || b[1].len != b[0].len || b[2].len != b[0].len || (kalbach && b[3].len != b[0].len)
        || b[5].len != b[4].len || m < 1) {
        PyErr_SetString(PyExc_ValueError, "native heating: inconsistent unit-base buffers");
        goto done;
    }
    terms = PyMem_Malloc(sizeof(double) * (size_t)((n - 1) * m));
    dnorm = PyMem_Malloc(sizeof(double) * (size_t)(n - 1));
    if (!terms || !dnorm) { PyErr_NoMemory(); goto done; }
    double integral;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n - 1; ++i) {
        double dx = x[i + 1] - x[i];
        dnorm[i] = flat ? dx * p[i] : (dx * (p[i] + p[i + 1])) / 2;
        double lm = qa + (qw * (x[i] - a)) / width, rm = qa + (qw * (x[i + 1] - a)) / width;
        double left = sqrt(np_maximum(lm, 0)), right = sqrt(np_maximum(rm, 0));
        double safe_dx = np_maximum(dx, 1e-300);
        for (Py_ssize_t k = 0; k < m; ++k) {
            double r = (left + right) / 2 + ((right - left) / 2) * nodes[k];
            double original = a + (((r * r) - qa) * width) / qw;
            double frac = (original - x[i]) / safe_dx;
            double density = flat ? p[i] : p[i] + (p[i + 1] - p[i]) * frac;
            double angle;
            if (kalbach) {
                double rr = flat ? ang[i] : ang[i] + (ang[i + 1] - ang[i]) * frac;
                double aa = flat ? sl[i] : sl[i] + (sl[i + 1] - sl[i]) * frac;
                double langevin;
                if (fabs(aa) > .01) langevin = 1 / crt_tanh(aa) - 1 / aa;
                else langevin = aa * (1. / 3 + aa * aa * (-1. / 45 + aa * aa * (2. / 945 - aa * aa / 4725)));
                angle = rr * langevin;
            } else {
                angle = flat ? ang[i] : ang[i] + (ang[i + 1] - ang[i]) * frac;
            }
            double t = (right - left) / 2;
            t *= ww[k]; t *= 2; t *= r; t *= r; t *= width; t /= qw; t *= density; t *= angle;
            terms[i * m + k] = t;
        }
    }
    integral = numpy_total(terms, (n - 1) * m) / numpy_total(dnorm, n - 1);
    Py_END_ALLOW_THREADS
    result = PyFloat_FromDouble(integral);
 done:
    PyMem_Free(terms);
    PyMem_Free(dnorm);
    for (int k = 0; k < 6; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return result;
}

static PyMethodDef methods[] = {
    {"unit_base_integral", unit_base_integral, METH_VARARGS,
     "One quadrature of the unit-base centre-of-mass cross moment, bit for bit the NumPy reference."},
    {"points_bin", points_bin, METH_VARARGS,
     "_MomentumBins.add of points with no variance, bit for bit np.bincount."},
    {"recoil_rows", recoil_rows, METH_VARARGS,
     "The one-particle recoil damage rows of _recoilDamage, bit for bit the NumPy reference."},
    {"momenta_damage", momenta_damage, METH_VARARGS,
     "The damage of the residual over two binned momenta added over the azimuth, bit for bit the NumPy reference."},
    {"momenta_bin", momenta_bin, METH_VARARGS,
     "Two binned momenta added over the azimuth and binned again, bit for bit the NumPy reference."},
    {"twobody_chunk_joint", twobody_chunk_joint, METH_VARARGS,
     "twobody_chunk for MT444 and two displacement models at once (3-tuple), bit for bit the NumPy reference."},
    {"emission_chunk_joint", emission_chunk_joint, METH_VARARGS,
     "emission_chunk for MT444 and two displacement models at once (3-tuple), bit for bit the NumPy reference."},
    {"twobody_chunk", twobody_chunk, METH_VARARGS,
     "One angular quadrature of the two-body recoil damage, bit for bit the NumPy reference."},
    {"emission_chunk", emission_chunk, METH_VARARGS,
     "One chunk of the tabulated emission-damage quadrature, bit for bit the NumPy reference."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_heating_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__heating_native(void)
{
#ifdef _WIN32
    HMODULE ucrt = GetModuleHandleW(L"ucrtbase.dll");
    if (!ucrt) ucrt = LoadLibraryW(L"ucrtbase.dll");
    FARPROC found = ucrt ? GetProcAddress(ucrt, "pow") : NULL;
    if (!found) {
        PyErr_SetString(PyExc_ImportError, "native heating: the Universal CRT's pow is not available");
        return NULL;
    }
    crt_pow = (pow_function)(void (*)(void))found;
    FARPROC root = GetProcAddress(ucrt, "cbrt");
    if (!root) {
        PyErr_SetString(PyExc_ImportError, "native heating: the Universal CRT's cbrt is not available");
        return NULL;
    }
    crt_cbrt = (cbrt_function)(void (*)(void))root;
    FARPROC hyperbolic = GetProcAddress(ucrt, "tanh");
    if (!hyperbolic) {
        PyErr_SetString(PyExc_ImportError, "native heating: the Universal CRT's tanh is not available");
        return NULL;
    }
    crt_tanh = (cbrt_function)(void (*)(void))hyperbolic;
#endif
    PyObject *mod = PyModule_Create(&module);
    if (mod && PyModule_AddIntConstant(mod, "API_VERSION", 6) < 0) { Py_DECREF(mod); return NULL; }
    return mod;
}
