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
    double out = crt_pow(ep, 1. / 6.);
    out *= 3.4008;
    double term = crt_pow(ep, 0.75);
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
    double out = crt_pow(ep, 1. / 6.);
    out *= 3.4008;
    double term = crt_pow(ep, 0.75);
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
    double T = crt_pow(ep, 1. / 6.);
    T *= 3.4008;
    double term = crt_pow(ep, 0.75);
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

static PyMethodDef methods[] = {
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
#endif
    PyObject *mod = PyModule_Create(&module);
    if (mod && PyModule_AddIntConstant(mod, "API_VERSION", 3) < 0) { Py_DECREF(mod); return NULL; }
    return mod;
}
