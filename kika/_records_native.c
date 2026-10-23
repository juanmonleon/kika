/* The hot loops of kika._records in C: reading and writing 11-column fields.
 *
 * Every conversion goes through CPython's own routines, PyOS_string_to_double
 * (what float() uses) and PyOS_double_to_string (what f"{x:.6e}" uses), so the
 * doubles read and the text written are those of the Python functions bit for
 * bit, whatever the platform's printf and strtod do. The C code only replaces
 * the Python-level work around them: slicing, stripping, matching the ENDF
 * float grammar, choosing between the scientific and the plain-decimal field.
 *
 * Anything outside the common grammar -- an explicit E exponent, a blank inside
 * a number, inf or nan, an identification field that is not plain digits, an
 * argument that is not a number -- is not decided here: each function calls
 * the Python definition it replaces, which kika._records hands over with
 * install() and which stays the definition. kika/tests/test_records_native.py
 * holds the two to the same results, and the equivalence sweep to every line of
 * the three major libraries.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

#define API_VERSION 1

static PyObject NOT_DECIDED_OBJ;
#define NOT_DECIDED (&NOT_DECIDED_OBJ) /* the Python code must decide */

/* ------------------------------------------------------------------ reading */

static int is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* 1: value parsed into *out (None when blank); 0: needs the Python path. */
static int parse_field(const char *s, Py_ssize_t n, PyObject **out)
{
    Py_ssize_t a = 0, b = n;
    while (a < b && is_space(s[a])) a++;
    while (b > a && is_space(s[b - 1])) b--;
    if (a == b) { Py_INCREF(Py_None); *out = Py_None; return 1; }

    /* sign, digits, [ '.' digits ], [ sign digits ]: nothing else is common */
    char buf[64];
    Py_ssize_t len = b - a;
    if (len > 40) return 0;
    const char *t = s + a;
    Py_ssize_t i = 0, k = 0;
    int mant_digits = 0, has_dot = 0, exp_digits = 0;
    if (t[i] == '+' || t[i] == '-') buf[k++] = t[i++];
    while (i < len && is_digit(t[i])) { buf[k++] = t[i++]; mant_digits++; }
    if (i < len && t[i] == '.') { buf[k++] = t[i++]; has_dot = 1;
        while (i < len && is_digit(t[i])) { buf[k++] = t[i++]; mant_digits++; } }
    if (i < len && (t[i] == '+' || t[i] == '-')) {
        /* The glued ENDF exponent: only after a decimal point, as the Python
           fast path requires; '1+5' without one goes to the Python search. */
        if (!has_dot) return 0;
        buf[k++] = 'e';
        buf[k++] = t[i++];
        while (i < len && is_digit(t[i])) { buf[k++] = t[i++]; exp_digits++; }
        if (exp_digits == 0) return 0;
        if (mant_digits == 0) { Py_INCREF(Py_None); *out = Py_None; return 1; } /* ".+3" */
    }
    if (i != len || mant_digits == 0) return 0;
    buf[k] = '\0';
    char *end = NULL;
    double v = PyOS_string_to_double(buf, &end, NULL);
    if (v == -1.0 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
    if (end == NULL || *end != '\0') return 0;
    if (!isfinite(v)) return 0;
    if (v == floor(v)) *out = PyLong_FromDouble(v);
    else *out = PyFloat_FromDouble(v);
    return *out != NULL ? 1 : -1;
}

/* Columns of an identification field: blank, or digits with an optional sign. */
static int plain_int_field(const char *s, Py_ssize_t n)
{
    Py_ssize_t a = 0, b = n;
    while (a < b && s[a] == ' ') a++;
    while (b > a && s[b - 1] == ' ') b--;
    if (a == b) return 1;
    if (s[a] == '-' || s[a] == '+') a++;
    if (a == b) return 0;
    for (Py_ssize_t i = a; i < b; i++) if (!is_digit(s[i])) return 0;
    return 1;
}

static PyObject *parse_record_values_fast(PyObject *arg)
{
    if (!PyUnicode_Check(arg)) { return NOT_DECIDED; }
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(arg, &n);
    if (s == NULL) { PyErr_Clear(); return NOT_DECIDED; }
    if (PyUnicode_GET_LENGTH(arg) != n) { return NOT_DECIDED; } /* not ASCII */
    if (n >= 75) {
        if (!plain_int_field(s + 66, 4) || !plain_int_field(s + 70, 2) || !plain_int_field(s + 72, 3)) {
            return NOT_DECIDED;
        }
    }
    if (n >= 80 && !plain_int_field(s + 75, 5)) { return NOT_DECIDED; }
    PyObject *tuple = PyTuple_New(6);
    if (tuple == NULL) return NULL;
    if (n < 66) {
        for (int i = 0; i < 6; i++) { Py_INCREF(Py_None); PyTuple_SET_ITEM(tuple, i, Py_None); }
        return tuple;
    }
    for (int i = 0; i < 6; i++) {
        PyObject *v = NULL;
        int r = parse_field(s + 11 * i, 11, &v);
        if (r == 1) { PyTuple_SET_ITEM(tuple, i, v); continue; }
        Py_DECREF(tuple);
        if (r < 0) return NULL;
        return NOT_DECIDED;
    }
    return tuple;
}

static PyObject *parse_number_fast(PyObject *arg)
{
    if (!PyUnicode_Check(arg)) { return NOT_DECIDED; }
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(arg, &n);
    if (s == NULL) { PyErr_Clear(); return NOT_DECIDED; }
    if (PyUnicode_GET_LENGTH(arg) != n) { return NOT_DECIDED; }
    PyObject *v = NULL;
    int r = parse_field(s, n, &v);
    if (r == 1) return v;
    if (r < 0) return NULL;
    return NOT_DECIDED;
}

/* ------------------------------------------------------------------ writing */

static int mantissa_precision(int exponent)
{
    int e = exponent < 0 ? -exponent : exponent;
    return e < 10 ? 6 : (e < 100 ? 5 : 4);
}

/* PyOS_double_to_string into buf; 0 on success. */
static int dtos(double v, char code, int prec, char *buf, size_t size)
{
    char *s = PyOS_double_to_string(v, code, prec, 0, NULL);
    if (s == NULL) return -1;
    size_t len = strlen(s);
    if (len + 1 > size) { PyMem_Free(s); return -1; }
    memcpy(buf, s, len + 1);
    PyMem_Free(s);
    return 0;
}

static int stod(const char *s, double *out)
{
    char *end = NULL;
    double v = PyOS_string_to_double(s, &end, NULL);
    if (v == -1.0 && PyErr_Occurred()) return -1;
    if (end == NULL || *end != '\0') { PyErr_SetString(PyExc_ValueError, "unreadable field"); return -1; }
    *out = v;
    return 0;
}

/* _scientific_parts: mantissa text (no sign) and exponent of the 11-column field. */
static int scientific_parts(double abs_val, char *mant, size_t msize, int *exponent)
{
    int prec = mantissa_precision((int)floor(log10(abs_val)));
    int seen_prec[3] = {0, 0, 0}, seen_exp[3] = {0, 0, 0}, nseen = 0;
    char buf[64];
    for (;;) {
        if (dtos(abs_val, 'e', prec, buf, sizeof buf)) return -1;
        char *e = strchr(buf, 'e');
        if (e == NULL) return -1;
        *e = '\0';
        int exp = atoi(e + 1);
        if (mantissa_precision(exp) == prec) {
            if (strlen(buf) + 1 > msize) return -1;
            strcpy(mant, buf);
            *exponent = exp;
            return 0;
        }
        int found = -1;
        for (int i = 0; i < nseen; i++) if (seen_prec[i] == prec) found = i;
        if (found >= 0) seen_exp[found] = exp;
        else if (nseen < 3) { seen_prec[nseen] = prec; seen_exp[nseen] = exp; nseen++; }
        prec = mantissa_precision(exp);
        for (int i = 0; i < nseen; i++) {
            if (seen_prec[i] == prec) {
                int carried = seen_exp[i] > exp ? seen_exp[i] : exp;
                int places = mantissa_precision(carried);
                mant[0] = '1'; mant[1] = '.';
                for (int j = 0; j < places; j++) mant[2 + j] = '0';
                mant[2 + places] = '\0';
                *exponent = carried;
                return 0;
            }
        }
    }
}

static void without_leading_zero(char *s)
{
    if (s[0] == '0' && s[1] == '.') memmove(s, s + 1, strlen(s));
    else if (s[0] == '-' && s[1] == '0' && s[2] == '.') memmove(s + 1, s + 2, strlen(s + 1));
}

static void rjust11(const char *s, char *out)
{
    size_t len = strlen(s);
    size_t pad = len < 11 ? 11 - len : 0;
    memset(out, ' ', pad);
    memcpy(out + pad, s, len + 1);
}

/* format_endf_fields for one finite non-zero value: the closest 11-column field. */
static int format_one(double value, char *out)
{
    if (value == 0) { strcpy(out, " 0.000000+0"); return 0; }
    double abs_val = value < 0 ? -value : value;
    char mant[32];
    int exponent;
    if (scientific_parts(abs_val, mant, sizeof mant, &exponent)) return -1;
    const char *sign_char = value < 0 ? "-" : " ";
    char exp_sign = exponent >= 0 ? '+' : '-';
    int aexp = exponent < 0 ? -exponent : exponent;
    char legacy[40], probe[48];
    snprintf(legacy, sizeof legacy, "%s%s%c%d", sign_char, mant, exp_sign, aexp);
    snprintf(probe, sizeof probe, "%s%se%c%d", value < 0 ? "-" : "", mant, exp_sign, aexp);
    double back;
    if (stod(probe, &back)) return -1;
    double legacy_error = fabs(back - value);
    if (legacy_error == 0.0) { strcpy(out, legacy); return 0; }

    int digits = (int)floor(log10(abs_val)) + 1;
    int sign = value < 0 ? 1 : 0;
    char best[40];
    strcpy(best, legacy);
    double best_error = legacy_error;
    int ndrops = digits <= 0 ? 2 : 1;
    for (int di = 0; di < ndrops; di++) {
        int drop = di == 1;
        int d = drop ? (digits > 0 ? digits : 0) : (digits > 1 ? digits : 1);
        if (d + sign > 11) continue;
        int places = 11 - d - sign - 1;
        if (places < 0) places = 0;
        char fixed[64], shorter[64];
        while (places >= 0) {
            if (dtos(value, 'f', places, fixed, sizeof fixed)) return -1;
            if (drop) without_leading_zero(fixed);
            if (strlen(fixed) <= 11) {
                double named;
                if (stod(fixed, &named)) return -1;
                while (places > 0) {
                    if (dtos(value, 'f', places - 1, shorter, sizeof shorter)) return -1;
                    if (drop) without_leading_zero(shorter);
                    double v2;
                    if (stod(shorter, &v2)) return -1;
                    if (v2 != named) break;
                    strcpy(fixed, shorter);
                    places--;
                }
                double error = fabs(named - value);
                if (error < best_error) { rjust11(fixed, best); best_error = error; }
                break;
            }
            places--;
        }
    }
    strcpy(out, best);
    return 0;
}

/* 1 with *v set; 0 when the Python code must take the value (not a float or int). */
/* The Python definitions, called whenever the C code does not decide. */
static PyObject *py_parse_number, *py_parse_record_values, *py_format_fields, *py_format_number;

static PyObject *fallback(PyObject *callable, PyObject *args, PyObject *kwargs)
{
    if (callable == NULL) { PyErr_SetString(PyExc_RuntimeError, "kika._records_native is not installed"); return NULL; }
    return PyObject_Call(callable, args, kwargs);
}

static PyObject *fallback1(PyObject *callable, PyObject *arg)
{
    if (callable == NULL) { PyErr_SetString(PyExc_RuntimeError, "kika._records_native is not installed"); return NULL; }
    return PyObject_CallOneArg(callable, arg);
}

static PyObject *native_parse_number(PyObject *self, PyObject *arg)
{
    PyObject *v = parse_number_fast(arg);
    if (v != NOT_DECIDED) return v;
    return fallback1(py_parse_number, arg);
}

static PyObject *native_parse_record_values(PyObject *self, PyObject *arg)
{
    PyObject *v = parse_record_values_fast(arg);
    if (v != NOT_DECIDED) return v;
    return fallback1(py_parse_record_values, arg);
}

/* format_endf_fields(values, data_format=None) */
static PyObject *native_format_fields(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 1 || nargs > 2 || (kwargs != NULL && PyDict_GET_SIZE(kwargs) > 0 &&
        !(PyDict_GET_SIZE(kwargs) == 1 && nargs == 1 && PyDict_GetItemString(kwargs, "data_format") != NULL)))
        return fallback(py_format_fields, args, kwargs);
    PyObject *seq = PySequence_Fast(PyTuple_GET_ITEM(args, 0), "values must be iterable");
    if (seq == NULL) { PyErr_Clear(); return fallback(py_format_fields, args, kwargs); }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    PyObject *list = PyList_New(n);
    if (list == NULL) { Py_DECREF(seq); return NULL; }
    char out[48];
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = items[i];
        double v;
        if (PyFloat_CheckExact(item)) v = PyFloat_AS_DOUBLE(item);
        else {
            /* float(value), as the Python loop does: numpy scalars, ints, strings */
            PyObject *f = PyNumber_Float(item);
            if (f == NULL) { PyErr_Clear(); v = NAN; }
            else { v = PyFloat_AS_DOUBLE(f); Py_DECREF(f); }
        }
        if (!isfinite(v) || format_one(v, out)) {
            /* the Python loop raises the error the value deserves */
            PyErr_Clear();
            Py_DECREF(list); Py_DECREF(seq);
            return fallback(py_format_fields, args, kwargs);
        }
        PyObject *s = PyUnicode_FromString(out);
        if (s == NULL) { Py_DECREF(list); Py_DECREF(seq); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    Py_DECREF(seq);
    return list;
}

/* format_endf_number(value, width=11): the 11-column field of a finite non-zero
   float or int here, everything else (None, zero, other widths, other types) in Python. */
static PyObject *native_format_number(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (PyTuple_GET_SIZE(args) == 1 && (kwargs == NULL || PyDict_GET_SIZE(kwargs) == 0)) {
        PyObject *item = PyTuple_GET_ITEM(args, 0);
        double v = 0.0;
        int ok = 0;
        if (PyFloat_CheckExact(item)) { v = PyFloat_AS_DOUBLE(item); ok = 1; }
        else if (PyLong_CheckExact(item)) {
            v = PyLong_AsDouble(item);
            if (v == -1.0 && PyErr_Occurred()) PyErr_Clear(); else ok = 1;
        }
        char out[48];
        if (ok && v != 0 && isfinite(v)) {
            if (format_one(v, out) == 0) return PyUnicode_FromString(out);
            PyErr_Clear();
        }
    }
    return fallback(py_format_number, args, kwargs);
}

static PyObject *native_install(PyObject *self, PyObject *args)
{
    PyObject *a, *b, *c, *d;
    if (!PyArg_ParseTuple(args, "OOOO", &a, &b, &c, &d)) return NULL;
    Py_XSETREF(py_parse_number, Py_NewRef(a));
    Py_XSETREF(py_parse_record_values, Py_NewRef(b));
    Py_XSETREF(py_format_fields, Py_NewRef(c));
    Py_XSETREF(py_format_number, Py_NewRef(d));
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"install", native_install, METH_VARARGS,
     "install(parse_number, parse_record_values, format_endf_fields, format_endf_number): "
     "the Python definitions every function below falls back to."},
    {"parse_number", native_parse_number, METH_O, "kika._records.parse_number, in C where it can."},
    {"parse_record_values", native_parse_record_values, METH_O,
     "kika._records.parse_record_values, in C where it can."},
    {"format_endf_fields", (PyCFunction)(void (*)(void))native_format_fields, METH_VARARGS | METH_KEYWORDS,
     "kika._records.format_endf_fields, in C where it can."},
    {"format_endf_number", (PyCFunction)(void (*)(void))native_format_number, METH_VARARGS | METH_KEYWORDS,
     "kika._records.format_endf_number, in C where it can."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_records_native",
    "ENDF field reading and writing in C, through CPython's own conversions.", -1, methods
};

PyMODINIT_FUNC PyInit__records_native(void)
{
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "API_VERSION", API_VERSION) < 0) return NULL;
    return m;
}
