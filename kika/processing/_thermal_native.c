/* Thermal incoherent inelastic scattering, NJOY THERMR's calcem/sigl/sig/terpq
 * (thermr.f90, NJOY2016) ported line for line, for kika.processing.thermal.
 * iinc = 2 is the tabulated S(alpha,beta); iinc = 1 the free gas (sig label 200).
 *
 * One call computes one incident energy: the adaptive outgoing-energy grid,
 * sigma(E->E') and the nbin equiprobable cosines at each E', and the
 * integrated cross section, exactly as calcem writes them before ACER. The
 * control flow keeps THERMR's labels as gotos so it can be read beside the
 * Fortran. Float64 only, strict IEEE: no fast-math.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifdef __FAST_MATH__
#error "the thermal kernel requires strict floating-point arithmetic"
#endif

/* Python's float ** is the C runtime's pow: the Universal CRT's (ucrtbase.dll) on
 * Windows, looked up at import because MinGW links its own; pow(x, 2) is not x*x. */
typedef double (*pow_function)(double, double);
static pow_function crt_pow = pow;

/* sigfig's powers of ten, P10[k + P10_SHIFT] = pow(10.0, k), filled at import
 * with the very pow sigfig used to call: the same bits by construction, without
 * a libm call per rounding. The exponent goes through a volatile so that the
 * compiler cannot fold the call at compile time with a different rounding. */
#define P10_SHIFT 400
static double P10[2 * P10_SHIFT + 1];
static void fill_p10(void) {
    for (int k = -P10_SHIFT; k <= P10_SHIFT; k++) {
        volatile double exponent = (double)k;
        P10[k + P10_SHIFT] = pow(10.0, exponent);
    }
}
static double p10(int k) {
    return (k >= -P10_SHIFT && k <= P10_SHIFT) ? P10[k + P10_SHIFT] : pow(10.0, k);
}

#define SABFLG (-225.0)
#define IMAX 20
#define NLMAX 65

typedef struct {
    double tev, tevz, az, sb, sb2, az2, teff, teff2, cliq;
    int lat, lasym, iinc;
    Py_ssize_t na, nb;
    const double *alpha, *beta, *sab;   /* sab[ia + na*ib], 0-based */
} Ctx;

#define SAB(c, ia, ib) ((c)->sab[(ia) + (c)->na * (ib)])

/* util.f90 sigfig */
static double sigfig(double x, int ndig, int idig) {
    double xx = 0.0;
    if (x != 0.0) {
        double aa = log10(fabs(x));
        int ipwr = (int)aa;
        if (aa < 0) ipwr = ipwr - 1;
        ipwr = ndig - 1 - ipwr;
        double v = x * p10(ipwr) + p10(ndig - 11);
        long long ii = (long long)(v >= 0 ? floor(v + 0.5) : -floor(-v + 0.5));
        long long lim = 1;
        for (int k = 0; k < ndig; k++) lim *= 10;
        if (ii >= lim) { ii = ii / 10; ipwr = ipwr - 1; }
        ii = ii + idig;
        xx = (double)ii * p10(-ipwr);
    }
    return xx * 1.0000000000001;
}

/* endf.f90 terp1, laws 2 (lin-lin) and 3 (y linear in ln x) */
static double terp1(double x1, double y1, double x2, double y2, double x, int law) {
    if (x2 == x1) return y1;
    if (law == 3) {
        if (x1 <= 0 || x2 <= 0 || x <= 0) return y1;
        return y1 + (y2 - y1) * log(x / x1) / log(x2 / x1);
    }
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

static double terpq(double x1, double y1, double x2, double y2,
                    double x3, double y3, double x) {
    double y;
    if (x < x1) {
        if (y1 > y2) y = y1;
        else y = terp1(x1, y1, x2, y2, x, 3);
    } else if (x > x3) {
        if (y3 > y2) y = y3;
        else y = terp1(x2, y2, x3, y3, x, 2);
    } else if (fabs(y1 - y2) > 2.0 || fabs(y2 - y3) > 2.0) {
        if (x < x2) y = terp1(x1, y1, x2, y2, x, 2);
        else y = terp1(x2, y2, x3, y3, x, 2);
    } else {
        double b = (y2 - y1) * (x3 - x1) / ((x2 - x1) * (x3 - x2));
        b = b - (y3 - y1) * (x2 - x1) / ((x3 - x1) * (x3 - x2));
        double c = (y3 - y1) / ((x3 - x1) * (x3 - x2));
        c = c - (y2 - y1) / ((x2 - x1) * (x3 - x2));
        y = y1 + b * (x - x1) + c * (x - x1) * (x - x1);
    }
    if (y < SABFLG) y = SABFLG;
    return y;
}

/* thermr.f90 sig: iinc=2 (tabulated S(alpha,beta)) or iinc=1 (free gas) */
static double sig(const Ctx *c, double e, double ep, double u) {
    const double sigmin = 1e-10, amin = 1e-6, test1 = 0.2, test2 = 30.0;
    double tev = c->tev, rtev = 1.0 / tev;
    double bb = (ep - e) * rtev;
    double a = (e + ep - 2 * u * sqrt(e * ep)) / (c->az * tev);
    if (c->iinc != 1 && a < amin) a = amin;
    double sigc = sqrt(ep / e) * rtev / 2;
    double b = fabs(bb);
    double cc = sqrt(4 * M_PI);
    double s, s1, s2, s3, bbb, arg, out;
    Py_ssize_t na = c->na, nb = c->nb, ia = 0, ib = 0;
    if (c->iinc == 1) {
        /* label 200: free gas, sb = ((az+1)/az)**2 per unit free cross section */
        if(ep==0) return 0;
        if(a<=0) return bb==0 ? INFINITY : 0;
        s = 0;
        arg = (a + bb) * (a + bb) / (4 * a);
        if (-arg > SABFLG) s = exp(-arg) / (cc * sqrt(a));
        out = sigc * c->sb * s;
        return out;
    }
    if (c->lat == 1) { b = b * tev / c->tevz; a = a * tev / c->tevz; }
    if (a > c->alpha[na - 1]) goto l170;
    if (c->lasym == 1) {
        double bbm = bb;
        if (c->lat == 1) bbm = bb * tev / c->tevz;
        if (bbm > c->beta[nb - 1]) goto l170;
        if (bbm < c->beta[0]) goto l170;
    } else {
        if (b > c->beta[nb - 1]) goto l170;
    }
    bbb = b;
    if (c->lasym == 1 && bb < 0) bbb = -b;
    /* THERMR scans for the first i in [0, n-2] with x < grid[i+1], else n-2 (0 when
     * n < 2). The grids ascend, so the test is monotone and a bisection finds the
     * same i in log(n) steps instead of ~n/2. */
    {
        Py_ssize_t lo = 0, hi = nb - 2, mid;
        while (lo < hi) { mid = (lo + hi) / 2; if (bbb < c->beta[mid + 1]) hi = mid; else lo = mid + 1; }
        ib = lo;
        lo = 0; hi = na - 2;
        while (lo < hi) { mid = (lo + hi) / 2; if (a < c->alpha[mid + 1]) hi = mid; else lo = mid + 1; }
        ia = lo;
    }
    if (c->cliq == 0.0 || a >= c->alpha[0]) goto l150;
    if (c->lasym == 1) goto l150;
    if (b > test1) goto l150;
    s = SAB(c, 0, 0) + log(c->alpha[0] / a) / 2 - c->cliq * b * b / a;
    if (s < SABFLG) s = SABFLG;
    goto l160;
l150:
    if (a * c->az < test2 && b < test2) goto l155;
    if (SAB(c, ia, ib) <= SABFLG) goto l170;
    if (SAB(c, ia + 1, ib) <= SABFLG) goto l170;
    if (SAB(c, ia, ib + 1) <= SABFLG) goto l170;
    if (SAB(c, ia + 1, ib + 1) <= SABFLG) goto l170;
l155:
    if (ia + 1 == na - 1) ia = ia - 1;
    if (ib + 1 == nb - 1) ib = ib - 1;
    s1 = terpq(c->alpha[ia], SAB(c, ia, ib), c->alpha[ia + 1], SAB(c, ia + 1, ib),
               c->alpha[ia + 2], SAB(c, ia + 2, ib), a);
    s2 = terpq(c->alpha[ia], SAB(c, ia, ib + 1), c->alpha[ia + 1], SAB(c, ia + 1, ib + 1),
               c->alpha[ia + 2], SAB(c, ia + 2, ib + 1), a);
    s3 = terpq(c->alpha[ia], SAB(c, ia, ib + 2), c->alpha[ia + 1], SAB(c, ia + 1, ib + 2),
               c->alpha[ia + 2], SAB(c, ia + 2, ib + 2), a);
    s = terpq(c->beta[ib], s1, c->beta[ib + 1], s2, c->beta[ib + 2], s3, bbb);
l160:
    out = 0;
    if (s - bb / 2 > SABFLG) out = exp(s - bb / 2);
    out = sigc * c->sb * out;
    if (out < sigmin) out = 0;
    return out;
l170:
    if (c->lat == 1) { b = b * c->tevz * rtev; a = a * c->tevz * rtev; }
    s = 0;
    arg = (a - b) * (a - b) * tev / (4 * a * c->teff) + (b + bb) / 2;
    if (-arg > SABFLG) s = exp(-arg) / (cc * sqrt(a * c->teff * rtev));
    out = sigc * c->sb * s;
    if (c->sb2 > 0) {
        a = a * c->az / c->az2;
        arg = (a - b) * (a - b) * tev / (4 * a * c->teff2) + (b + bb) / 2;
        s2 = 0;
        if (-arg > SABFLG) s2 = exp(-arg) / (cc * sqrt(a * c->teff2 * rtev));
        out = out + sigc * c->sb2 * s2;
    }
    if (out < sigmin) out = 0;
    return out;
}

/* thermr.f90 sigl with nlin < 0: s[0] = sigma(E->E'), s[1..nl-1] equiprobable cosines.
 * Arrays are 1-based as in the Fortran (index 0 unused). */
/* Free-gas angular integration in u=sqrt(alpha). The Jacobian removes the
 * integrable 1/sqrt(alpha) singularity without an alpha floor. */
static double fg_erfcx(double x) {
    if (x < 25.0) return exp(x*x)*erfc(x);
    double t=1.0/(2*x*x), term=1.0, sum=1.0;
    for (int k=1;k<=8;k++) { term*=-(2*k-1)*t;sum+=term; }
    return sum/(sqrt(M_PI)*x);
}
static double fg_derf(double a,double b) {
    if (a>=0 && b>=0) return erfc(a)-erfc(b);
    if (a<=0 && b<=0) return erfc(-b)-erfc(-a);
    return erf(b)-erf(a);
}
static double fg_weight(double u,double beta) {
    if (u==0) return beta==0 ? 1.0 : 0.0;
    double p=(u+beta/u)/2;
    return exp(-p*p);
}
static const double fg_x8[4]={.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
static const double fg_w8[4]={.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
static double fg_quad(double a,double b,double beta,int moment,double ulo,double scale) {
    double mid=(a+b)/2,half=(b-a)/2,sum=0;
    for (int k=0;k<4;k++) for(int sign=-1;sign<=1;sign+=2) {
        double u=mid+sign*half*fg_x8[k];
        double value=fg_weight(u,beta);
        if (moment) value*=1-scale*(u-ulo)*(u+ulo);
        sum+=fg_w8[k]*value;
    }
    return sum*half;
}
static double fg_mass(double a,double b,double beta) {
    if (b<=a) return 0;
    if (beta==0) return sqrt(M_PI)*fg_derf(a/2,b/2);
    double pa=(a+beta/a)/2,pb=(b+beta/b)/2;
    double qa=(a-beta/a)/2,qb=(b-beta/b)/2;
    double one=fg_derf(pa,pb),two;
    if (beta<0) two=exp(-pa*pa)*fg_erfcx(qa)-exp(-pb*pb)*fg_erfcx(qb);
    else two=exp(-beta)*fg_derf(qa,qb);
    double value=sqrt(M_PI)*(one+two)/2;
    /* Narrow intervals and cancelling tails use a direct, smooth integral. */
    if (b-a<1e-5*fmax(1.0,b) || value<1e-8*(fabs(one)+fabs(two)))
        value=fg_quad(a,b,beta,0,0,0);
    return fmax(value,0.0);
}
static double fg_mean_integral(double a,double b,double beta,double ulo,double scale,int depth) {
    double coarse=fg_quad(a,b,beta,1,ulo,scale),mid=(a+b)/2;
    double fine=fg_quad(a,mid,beta,1,ulo,scale)+fg_quad(mid,b,beta,1,ulo,scale);
    double mass=fg_mass(a,b,beta);
    if (fabs(fine-coarse)<=1e-11*mass+1e-300 || depth>=20) return fine;
    return fg_mean_integral(a,mid,beta,ulo,scale,depth+1)+fg_mean_integral(mid,b,beta,ulo,scale,depth+1);
}
static void freegas_sigl(const Ctx *c,double e,double ep,int nl,double *s) {
    int bins=nl-1;
    for(int j=0;j<nl;j++)s[j]=0;
    if (ep<=0) {for(int j=1;j<=bins;j++)s[j]=-1+(2.0*j-1)/bins;return;}
    double root=sqrt(c->az*c->tev),se=sqrt(e),sp=sqrt(ep);
    double lo=fabs(sp-se)/root,hi=(sp+se)/root,beta=(ep-e)/c->tev;
    double mass=fg_mass(lo,hi,beta);
    if (!(mass>1e-290)) return;
    s[0]=c->sb*c->az/(4*e*sqrt(M_PI))*mass;
    double upper=hi,scale=c->az*c->tev/(2*sqrt(e*ep));
    for(int j=1;j<=bins;j++) {
        double target=mass*(1-(double)j/bins),a=lo,b=upper;
        if(j<bins) for(int it=0;it<48;it++) {
            double mid=(a+b)/2;
            if(fg_mass(lo,mid,beta)<target)a=mid;else b=mid;
        }
        double edge=j==bins ? lo : (a+b)/2;
        double area=fg_mass(edge,upper,beta);
        s[j]=area>0 ? fg_mean_integral(edge,upper,beta,lo,scale,0)/area : 1-scale*((edge+upper)/2-lo)*((edge+upper)/2+lo);
        s[j]=fmax(-1,fmin(1,s[j]));upper=edge;
    }
}

/* The points sigl's first pass accepts, in the order it accepts them. The second
 * pass of THERMR's sigl rebuilds the same stack from {-1, mu0, +1} and applies the
 * same tests in the same order, so it accepts the same points: replaying them gives
 * the same bins without evaluating sig again (half of all sig calls). A buffer that
 * could not grow falls back to THERMR's second pass. */
typedef struct { double *x, *y; Py_ssize_t n, cap; int failed; } Accepted;

static void accepted_push(Accepted *a, double x, double y) {
    if (a->failed) return;
    if (a->n == a->cap) {
        Py_ssize_t cap = a->cap ? 2 * a->cap : 256;
        double *nx = (double *)realloc(a->x, (size_t)cap * sizeof(double));
        if (!nx) { a->failed = 1; return; }
        a->x = nx;
        double *ny = (double *)realloc(a->y, (size_t)cap * sizeof(double));
        if (!ny) { a->failed = 1; return; }
        a->y = ny;
        a->cap = cap;
    }
    a->x[a->n] = x;
    a->y[a->n] = y;
    a->n++;
}

static void accepted_free(Accepted *a) { free(a->x); free(a->y); }

/* thermr.f90 sigl labels 160-250 over the recorded points: the bins of equal
 * probability and their mean cosines, as the second pass computes them. */
static void sigl_bins(int nl, double total, const Accepted *pts, double *s) {
    const double ytol = 0.001, half = 0.5, third = 0.333333333, sigmin = 1e-32, shade = 0.99999999;
    int nbin = nl - 1, j = 0, il;
    double rnbin = 1.0 / nbin, fract = total * rnbin, rfract = 1 / fract, sum = 0, gral = 0;
    double xl = -1, yl = pts->y[0], add, xil, xn = 0, f, rf, disc, yn, xbar, test, xi, yi;
    Py_ssize_t k, last = pts->n - 1;
    for (il = 1; il < nl; il++) s[il] = 0;
    for (k = 0; k <= last; k++) {
        xi = pts->x[k];
        yi = pts->y[k];
    r160:
        add = half * (yi + yl) * (xi - xl);
        if (xi == xl) goto r250;
        xil = 1 / (xi - xl);
        if (k == last && j == nbin - 1) goto r165;
        if (sum + add >= fract * shade && j < nbin - 1) goto r170;
        sum = sum + add;
        gral = gral + half * (yl * xi - yi * xl) * (xi + xl)
               + third * (yi - yl) * (xi * xi + xi * xl + xl * xl);
        goto r250;
    r165:
        xn = xi;
        j = j + 1;
        goto r190;
    r170:
        j = j + 1;
        if (yl < sigmin) goto r175;
        test = (fract - sum) * (yi - yl) / ((xi - xl) * yl * yl);
        if (fabs(test) > ytol) goto r175;
        xn = xl + (fract - sum) / yl;
        if (xn > xi) goto r180;
        if (xn >= xl && xn <= xi) goto r190;
        goto r180;
    r175:
        f = (yi - yl) * xil;
        rf = 1 / f;
        disc = (yl * rf) * (yl * rf) + 2 * (fract - sum) * rf;
        if (disc < 0) disc = fabs(disc);
        if (f > 0) xn = xl - (yl * rf) + sqrt(disc);
        if (f < 0) xn = xl - (yl * rf) - sqrt(disc);
        if (xn > xl && xn <= xi) goto r190;
        if (xn > xl && xn < (xi + ytol * (xi - xl))) goto r180;
    r180:
        xn = xi;
    r190:
        yn = yl + (yi - yl) * (xn - xl) * xil;
        gral = gral + (xn - xl) * (yl * half * (xn + xl)
               + (yi - yl) * xil * (-xl * half * (xn + xl)
               + third * (xn * xn + xn * xl + xl * xl)));
        xbar = gral * rfract;
        s[j] = xbar;
        xl = xn;
        yl = yn;
        sum = 0;
        gral = 0;
        if (j == nbin) return;
        if (xl < xi) goto r160;
    r250:
        xl = xi;
        yl = yi;
    }
}

static void sigl(const Ctx *c, double e, double ep, int nl, double tolin, double *s) {
    if(c->iinc==1){freegas_sigl(c,e,ep,nl,s);return;}

    const double xtol = 0.00001, ytol = 0.001, half = 0.5, third = 0.333333333;
    const double sigmin = 1e-32, eps = 1e-3, shade = 0.99999999;
    double x[IMAX + 1], y[IMAX + 1];
    double tol = half * tolin, b = fabs((ep - c->tev * 0 - e) / c->tev);
    double seep = 0, sum, xl, yl, ymax, xm, ym, yt, test, test2;
    int i, j, nbin, il;
    double rnbin, fract, rfract, gral, add, xil, xn = 0, f, rf, disc, yn, xbar;
    b = fabs((ep - e) / c->tev);
    if (c->lat == 1 && c->iinc == 2) b = b * c->tev / c->tevz;
    double s1bb = sqrt(1 + b * b);
    if (ep != 0) seep = 1 / sqrt(e * ep);
    Accepted accepted = {0};

    /* adaptive calculation of cross section */
    i = 3;
    sum = 0;
    x[3] = -1;
    xl = x[3];
    y[3] = sig(c, e, ep, x[3]);
    yl = y[3];
    if (ep == 0) x[2] = 0;
    if (ep != 0) x[2] = half * (e + ep - (s1bb - 1) * c->az * c->tev) * seep;
    if (fabs(x[2]) > 1 - eps) x[2] = 0.99;
    x[2] = sigfig(x[2], 8, 0);
    y[2] = sig(c, e, ep, x[2]);
    x[1] = +1;
    y[1] = sig(c, e, ep, x[1]);
    ymax = y[2];
    if (y[1] > ymax) ymax = y[1];
    if (y[3] > ymax) ymax = y[3];
    if (ymax < eps) ymax = eps;
l110:
    if (i == IMAX) goto l120;
    xm = half * (x[i - 1] + x[i]);
    xm = sigfig(xm, 8, 0);
    ym = half * (y[i - 1] + y[i]);
    yt = sig(c, e, ep, xm);
    test = tol * fabs(yt) + tol * ymax / 50;
    test2 = ym + ymax / 100;
    if (fabs(yt - ym) <= test && fabs(y[i - 1] - y[i]) <= test2 && (x[i - 1] - x[i]) < half) goto l120;
    if (x[i - 1] - x[i] < xtol) goto l120;
    i = i + 1;
    x[i] = x[i - 1];
    y[i] = y[i - 1];
    x[i - 1] = xm;
    y[i - 1] = yt;
    goto l110;
l120:
    accepted_push(&accepted, x[i], y[i]);
    sum = sum + half * (y[i] + yl) * (x[i] - xl);
    xl = x[i];
    yl = y[i];
    i = i - 1;
    if (i > 1) goto l110;
    if (i == 1) goto l120;
    s[0] = sum;
    if (sum > sigmin) goto l130;
    accepted_free(&accepted);
    for (il = 0; il < nl; il++) s[il] = 0;
    return;

l130:
    if (!accepted.failed && accepted.n > 0) {
        sigl_bins(nl, sum, &accepted, s);
        accepted_free(&accepted);
        return;
    }
    accepted_free(&accepted);
    nbin = nl - 1;
    rnbin = 1.0 / nbin;
    fract = sum * rnbin;
    rfract = 1 / fract;
    sum = 0;
    gral = 0;
    for (il = 1; il < nl; il++) s[il] = 0;
    j = 0;
    i = 3;
    x[3] = -1;
    xl = x[3];
    y[3] = sig(c, e, ep, x[3]);
    if (ep == 0) x[2] = 0;
    if (ep != 0) x[2] = half * (e + ep - (s1bb - 1) * c->az * c->tev) * seep;
    if (fabs(x[2]) > 1 - eps) x[2] = 0.99;
    x[2] = sigfig(x[2], 8, 0);
    y[2] = sig(c, e, ep, x[2]);
    x[1] = +1;
    y[1] = sig(c, e, ep, x[1]);
    ymax = y[1];
    if (y[2] > ymax) ymax = y[2];
    if (y[3] > ymax) ymax = y[3];
    if (ymax < eps) ymax = eps;
    /* yl carries over from the first pass's last point? No: THERMR does not
     * reset yl here; at this point yl is y(1) of the first pass (mu = +1). The
     * first panel of the second pass starts at xl = -1, so add() uses y(3)
     * through yl only after the first accepted point. Keep THERMR's value. */
l150:
    if (i == IMAX) goto l160;
    xm = half * (x[i - 1] + x[i]);
    xm = sigfig(xm, 8, 0);
    ym = half * (y[i - 1] + y[i]);
    yt = sig(c, e, ep, xm);
    test = tol * fabs(yt) + tol * ymax / 50;
    test2 = ym + ymax / 100;
    if (fabs(yt - ym) <= test && fabs(y[i - 1] - y[i]) <= test2 && (x[i - 1] - x[i]) < half) goto l160;
    if (x[i - 1] - x[i] < xtol) goto l160;
    i = i + 1;
    x[i] = x[i - 1];
    y[i] = y[i - 1];
    x[i - 1] = xm;
    y[i - 1] = yt;
    goto l150;
l160:
    add = half * (y[i] + yl) * (x[i] - xl);
    if (x[i] == xl) goto l250;
    xil = 1 / (x[i] - xl);
    if (i == 1 && j == nbin - 1) goto l165;
    if (sum + add >= fract * shade && j < nbin - 1) goto l170;
    sum = sum + add;
    gral = gral + half * (yl * x[i] - y[i] * xl) * (x[i] + xl)
           + third * (y[i] - yl) * (x[i] * x[i] + x[i] * xl + xl * xl);
    goto l250;
l165:
    xn = x[i];
    j = j + 1;
    goto l190;
l170:
    j = j + 1;
    if (yl < sigmin) goto l175;
    test = (fract - sum) * (y[i] - yl) / ((x[i] - xl) * yl * yl);
    if (fabs(test) > ytol) goto l175;
    xn = xl + (fract - sum) / yl;
    if (xn > x[i]) goto l180;
    if (xn >= xl && xn <= x[i]) goto l190;
    goto l180;  /* THERMR stops with 'no legal solution'; take the panel end */
l175:
    f = (y[i] - yl) * xil;
    rf = 1 / f;
    disc = (yl * rf) * (yl * rf) + 2 * (fract - sum) * rf;
    if (disc < 0) disc = fabs(disc);
    if (f > 0) xn = xl - (yl * rf) + sqrt(disc);
    if (f < 0) xn = xl - (yl * rf) - sqrt(disc);
    if (xn > xl && xn <= x[i]) goto l190;
    if (xn > xl && xn < (x[i] + ytol * (x[i] - xl))) goto l180;
l180:
    xn = x[i];
l190:
    yn = yl + (y[i] - yl) * (xn - xl) * xil;
    gral = gral + (xn - xl) * (yl * half * (xn + xl)
           + (y[i] - yl) * xil * (-xl * half * (xn + xl)
           + third * (xn * xn + xn * xl + xl * xl)));
    xbar = gral * rfract;
    s[j] = xbar;           /* s(j+1) in 1-based Fortran */
    xl = xn;
    yl = yn;
    sum = 0;
    gral = 0;
    if (j == nbin) return;
    if (xl < x[i]) goto l160;
l250:
    xl = x[i];
    yl = y[i];
    i = i - 1;
    if (i > 1) goto l150;
    if (i == 1) goto l160;
    return;
}

/* thermr.f90 calcem, one incident energy (labels 310-430). Writes up to cap
 * points (E', y[0..nl-1]) into xs/ys and returns the count; xsi gets the
 * integrated cross section. Points carry THERMR's sigfig rounding. */
static Py_ssize_t calcem_energy(const Ctx *c, double enow, int nl, double tol,
                                double *xs, double *ys, Py_ssize_t cap, double *xsiOut, int *overflow) {
    const double tolmin = c->iinc==1 ? 0.0 : 5e-7, em9 = 1e-9, half = 0.5, uumin = 1e-5;
    double x[IMAX + 1], y[NLMAX][IMAX + 1], yt[NLMAX];
    Py_ssize_t j = 0, jnz = 0;
    int i, il, k, jbeta, iskip = 0;
    double xsi = 0, ep, xm, ym, uu, uum, test, test2, xlast = 0, ylast = 0;
    Py_ssize_t nb = c->nb;
    /* Numerical floors apply only far below the reference checker’s 1%-of-peak
     * region; retain relative refinement throughout that region. */
    double freePeak=c->iinc==1 ? c->sb*c->az/(4*enow)*erf(sqrt(enow/(c->az*c->tev))) : 0;
    *overflow = 0;

#define STORE(jj, xv, yv) do { \
        if ((jj) - 1 >= cap) { *overflow = 1; return 0; } \
        xs[(jj) - 1] = (xv); \
        double y1_ = (yv)[0]; \
        ys[((jj) - 1) * nl] = (y1_ >= em9) ? sigfig(y1_, 9, 0) : sigfig(y1_, 8, 0); \
        for (int q_ = 1; q_ < nl; q_++) { \
            double v_ = sigfig((yv)[q_], 9, 0); \
            if (v_ > 1.0) v_ = 1.0; \
            if (v_ < -1.0) v_ = -1.0; \
            ys[((jj) - 1) * nl + q_] = v_; \
        } \
    } while (0)

    ep = 0;
    x[1] = ep;
    sigl(c, enow, ep, nl, tol, yt);
    for (il = 0; il < nl; il++) y[il][1] = yt[il];
    jbeta = -(int)nb;
    if (c->lasym > 0) jbeta = 1;
    j = 0;
    iskip = 0;
l311:
    x[2] = x[1];
    for (il = 0; il < nl; il++) y[il][2] = y[il][1];
l313:
    if (jbeta == 0) jbeta = 1;
    if (jbeta <= 0) {
        if (c->lat == 1) ep = enow - c->beta[-jbeta - 1] * c->tevz;
        else ep = enow - c->beta[-jbeta - 1] * c->tev;
        if (ep == enow) ep = sigfig(enow, 8, -1);
        else ep = sigfig(ep, 8, 0);
    } else {
        if (c->lat == 1) ep = enow + c->beta[jbeta - 1] * c->tevz;
        else ep = enow + c->beta[jbeta - 1] * c->tev;
        if (ep == enow) { ep = sigfig(enow, 8, +1); iskip = 1; }
        else ep = sigfig(ep, 8, 0);
    }
    if (ep > x[2]) goto l316;
    jbeta = jbeta + 1;
    goto l313;
l316:
    ep = sigfig(ep, 8, 0);
    x[1] = ep;
    sigl(c, enow, ep, nl, tol, yt);
    for (il = 0; il < nl; il++) y[il][1] = yt[il];
    i = 2;
l330:
    if (i == IMAX) goto l360;
    if (iskip == 1) { iskip = 0; goto l360; }
    if (half * (y[0][i - 1] + y[0][i]) * (x[i - 1] - x[i]) < tolmin) goto l360;
    xm = half * (x[i - 1] + x[i]);
    xm = sigfig(xm, 8, 0);
    if (xm <= x[i] || xm >= x[i - 1]) goto l360;
    sigl(c, enow, xm, nl, tol, yt);
    uu = 0;
    uum = 0;
    for (k = 0; k < nl; k++) {
        if(c->iinc==1 && k>0 && yt[0]<1e-12*freePeak) continue;
        ym = terp1(x[i], y[k][i], x[i - 1], y[k][i - 1], xm, 2);
        if (k > 0) { uu = uu + yt[k]; uum = uum + ym; }
        test = tol * fabs(yt[k]);
        if(c->iinc==1 && k==0) test+=tol*1e-8*freePeak;
        test2 = test;
        if (k > 0) test2 = tol;
        if (fabs(yt[k] - ym) > test2) goto l410;
    }
    test = 2 * tol * fabs(uu) + uumin;
    if (fabs(uu - uum) > test) goto l410;
l360:
    j = j + 1;
    if (j > 1) xsi = xsi + (x[i] - xlast) * (y[0][i] + ylast) * half;
    if (j == 3 && xsi < tolmin) j = 2;
    {
        double v[NLMAX];
        for (il = 0; il < nl; il++) v[il] = y[il][i];
        STORE(j, x[i], v);
    }
    xlast = x[i];
    ylast = y[0][i];
    if (ylast != 0) jnz = j;
    i = i - 1;
    if (i >= 2) goto l330;
    jbeta = jbeta + 1;
    if (jbeta <= (int)nb) goto l311;
    for (il = 0; il < nl; il++) y[il][i] = 0;
    goto l430;
l410:
    i = i + 1;
    x[i] = x[i - 1];
    x[i - 1] = xm;
    for (il = 0; il < nl; il++) { y[il][i] = y[il][i - 1]; y[il][i - 1] = yt[il]; }
    goto l330;
l430:
    j = j + 1;
    xsi = xsi + (x[i] - xlast) * (y[0][i] + ylast) * half;
    xsi = sigfig(xsi, 9, 0);
    {
        double v[NLMAX];
        for (il = 0; il < nl; il++) v[il] = y[il][i];
        STORE(j, x[i], v);
    }
    if (y[0][1] != 0) jnz = j;
    if (jnz < j) j = jnz + 1;
    *xsiOut = xsi;
    return j;
#undef STORE
}

static int buffer(PyObject *obj, Py_buffer *b, Py_ssize_t count, int writable) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, b, flags) < 0) return 0;
    if (b->ndim != 1 || b->itemsize != 8 || b->format == NULL || strcmp(b->format, "d") != 0 ||
        b->len < count * 8) {
        PyErr_SetString(PyExc_ValueError, "thermal kernel requires contiguous float64 buffers");
        return 0;
    }
    return 1;
}

static PyObject *calcem(PyObject *self, PyObject *args) {
    double enow, tev, tevz, az, sb, sb2, az2, teff, teff2, cliq, tol;
    int lat, lasym, nl, iinc = 2;
    Py_ssize_t na, nb, cap;
    PyObject *o[5];
    Py_buffer b[5] = {{0}};
    if (!PyArg_ParseTuple(args, "ddddddddddiiidnnnOOOOO|i", &enow, &tev, &tevz, &az, &sb, &sb2, &az2,
                          &teff, &teff2, &cliq, &lat, &lasym, &nl, &tol, &na, &nb, &cap,
                          &o[0], &o[1], &o[2], &o[3], &o[4], &iinc))
        return NULL;
    if (iinc != 1 && iinc != 2) {
        PyErr_SetString(PyExc_ValueError, "iinc is 1 (free gas) or 2 (S(alpha,beta))");
        return NULL;
    }
    if (nl < 2 || nl > NLMAX || na < 3 || nb < 3 || cap < 2) {
        PyErr_SetString(PyExc_ValueError, "invalid thermal kernel dimensions");
        return NULL;
    }
    Py_ssize_t sizes[5] = {na, nb, na * nb, cap, cap * nl};
    for (int q = 0; q < 5; q++) {
        if (!buffer(o[q], &b[q], sizes[q], q >= 3)) {
            for (int r = 0; r < 5; r++) if (b[r].obj) PyBuffer_Release(&b[r]);
            return NULL;
        }
    }
    Ctx c = {tev, tevz, az, sb, sb2, az2, teff, teff2, cliq, lat, lasym, iinc, na, nb,
             b[0].buf, b[1].buf, b[2].buf};
    double xsi = 0;
    int overflow = 0;
    Py_ssize_t n;
    Py_BEGIN_ALLOW_THREADS
    n = calcem_energy(&c, enow, nl, tol, b[3].buf, b[4].buf, cap, &xsi, &overflow);
    Py_END_ALLOW_THREADS
    for (int r = 0; r < 5; r++) if (b[r].obj) PyBuffer_Release(&b[r]);
    if (overflow) {
        PyErr_SetString(PyExc_MemoryError, "outgoing-energy grid exceeds the capacity given");
        return NULL;
    }
    return Py_BuildValue("nd", n, xsi);
}

static PyObject *py_sig(PyObject *self, PyObject *args) {
    double e, ep, u, tev, tevz, az, sb, sb2, az2, teff, teff2, cliq;
    int lat, lasym, iinc = 2;
    Py_ssize_t na, nb;
    PyObject *o[3];
    Py_buffer b[3] = {{0}};
    if (!PyArg_ParseTuple(args, "ddddddddddddiinnOOO|i", &e, &ep, &u, &tev, &tevz, &az, &sb, &sb2,
                          &az2, &teff, &teff2, &cliq, &lat, &lasym, &na, &nb, &o[0], &o[1], &o[2],
                          &iinc))
        return NULL;
    if (iinc != 1 && iinc != 2) {
        PyErr_SetString(PyExc_ValueError, "iinc is 1 (free gas) or 2 (S(alpha,beta))");
        return NULL;
    }
    Py_ssize_t sizes[3] = {na, nb, na * nb};
    for (int q = 0; q < 3; q++) {
        if (!buffer(o[q], &b[q], sizes[q], 0)) {
            for (int r = 0; r < 3; r++) if (b[r].obj) PyBuffer_Release(&b[r]);
            return NULL;
        }
    }
    Ctx c = {tev, tevz, az, sb, sb2, az2, teff, teff2, cliq, lat, lasym, iinc, na, nb,
             b[0].buf, b[1].buf, b[2].buf};
    double v = sig(&c, e, ep, u);
    for (int r = 0; r < 3; r++) if (b[r].obj) PyBuffer_Release(&b[r]);
    return PyFloat_FromDouble(v);
}

static PyObject *py_sigfig(PyObject *self, PyObject *args) {
    double x;
    int ndig, idig;
    if (!PyArg_ParseTuple(args, "dii", &x, &ndig, &idig)) return NULL;
    return PyFloat_FromDouble(sigfig(x, ndig, idig));
}

/* endf.f90 a11: the value an 11-column ENDF field written by NJOY reads back as.
 * Positive values in (0.1, 1e7): the extended 9-figure fixed form (8 decimals
 * below 1); everything else: 7, 6 or 5 significant figures by exponent width. */
static double endf11(double x) {
    char buf[64];
    if (x == 0.0) return 0.0;
    if (x > 0.1 && x < 0.999999999e7) {
        if (x < 0.999999999) snprintf(buf, sizeof buf, "%.8f", x);
        else snprintf(buf, sizeof buf, "%.8e", x);          /* 9 significant figures */
        return strtod(buf, NULL);
    }
    double ax = fabs(x);
    int n = (int)floor(log10(ax));
    int e = n < 0 ? -n : n;
    int digits = e < 10 ? 7 : (e < 100 ? 6 : 5);
    snprintf(buf, sizeof buf, "%.*e", digits - 1, x);
    return strtod(buf, NULL);
}

static PyObject *py_endf11(PyObject *self, PyObject *args) {
    PyObject *o;
    Py_buffer b = {0};
    if (!PyArg_ParseTuple(args, "O", &o)) return NULL;
    if (PyObject_GetBuffer(o, &b, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) return NULL;
    if (b.itemsize != 8 || b.format == NULL || strcmp(b.format, "d") != 0) {
        PyBuffer_Release(&b);
        PyErr_SetString(PyExc_ValueError, "endf11 needs a writable contiguous float64 array");
        return NULL;
    }
    double *v = b.buf;
    Py_ssize_t n = b.len / 8;
    for (Py_ssize_t i = 0; i < n; i++) v[i] = endf11(v[i]);
    PyBuffer_Release(&b);
    Py_RETURN_NONE;
}

/* THERMR's tpend then ACER's acesix (iwt = 2) on one incident energy, operation for
 * operation thermal._acer's reference loop (kept in the tests): the density over the
 * integrated cross section to 7 figures (6 below 1e-9), every field through NJOY's a11
 * writer, then the equiprobable-area thinning. Out: E', the density before
 * normalisation, each point's area and its clipped cosines; the caller normalises. */
typedef struct { double *x, *y, *f, *c; Py_ssize_t n, cap; int nang; } AceOut;

static int ace_push(AceOut *o, double x, double y, double f, const double *c) {
    if (o->n == o->cap) {
        Py_ssize_t cap = o->cap ? 2 * o->cap : 256;
        double *nx = realloc(o->x, cap * sizeof(double)), *ny, *nf, *nc;
        if (!nx) return 0;
        o->x = nx;
        if (!(ny = realloc(o->y, cap * sizeof(double)))) return 0;
        o->y = ny;
        if (!(nf = realloc(o->f, cap * sizeof(double)))) return 0;
        o->f = nf;
        if (!(nc = realloc(o->c, cap * o->nang * sizeof(double)))) return 0;
        o->c = nc;
        o->cap = cap;
    }
    o->x[o->n] = x; o->y[o->n] = y; o->f[o->n] = f;
    double *dst = o->c + o->n * o->nang;
    for (int j = 0; j < o->nang; j++)
        dst[j] = c[j] < -1.0 ? -1.0 : (c[j] > 1.0 ? 1.0 : c[j]);    /* np.clip */
    o->n++;
    return 1;
}

static int ace_energy(const double *xin, const double *yin, Py_ssize_t nep, int nl, double xsi,
                      int nang, AceOut *o) {
    double *x = malloc((nep ? nep : 1) * sizeof(double));
    double *y = malloc((nep ? nep : 1) * nl * sizeof(double));
    double *cos = malloc((nep ? nep : 1) * nang * sizeof(double));
    double *c = malloc((nang ? nang : 1) * sizeof(double));
    int ok = x && y && cos && c;
    if (!ok) goto done;
    double rx = 1.0 / xsi;
    for (Py_ssize_t i = 0; i < nep; i++) {
        x[i] = endf11(xin[i]);
        double v = yin[i * nl] * rx;
        y[i * nl] = endf11(v >= 1e-9 ? sigfig(v, 7, 0) : sigfig(v, 6, 0));
        for (int j = 1; j < nl; j++) y[i * nl + j] = endf11(yin[i * nl + j]);
        for (int j = 0; j < nang; j++) cos[i * nang + j] = y[i * nl + 1 + j];
    }
    if (nep >= 3) {
        memcpy(cos, cos + nang, nang * sizeof(double));
        memcpy(cos + (nep - 1) * nang, cos + (nep - 2) * nang, nang * sizeof(double));
    }
    const double eps = 1e-5;
    double xl = nep ? x[0] : 0, yl = nep ? y[0] : 0, total = 0.0, fract = 1.0;
    Py_ssize_t i = 0, k = 1;
    while (i < nep) {
        double xi = x[i], yi = y[i * nl], xn;
        i++;
        for (;;) {
            double add = (yi + yl) * (xi - xl) / 2;
            if (xi == xl) break;
            if (i == nep) {
                xn = xi;
                fract = total + add;
            } else {
                if (total + add > eps / 10) fract = total + add;
                if (!(total + add >= fract - fract / 10000)) { total += add; break; }
                if (total + add < fract + fract / 10000) xn = xi;
                else if (fabs(yi - yl) > (yi + yl) / 100000) {
                    double f = (yi - yl) / (xi - xl);
                    double disc = fabs(crt_pow(yl / f, 2.0) + 2 * (fract - total) / f);
                    xn = xl - yl / f + (f > 0 ? 1.0 : -1.0) * sqrt(disc);
                } else {
                    double a = xl + (fract - total) / yl;
                    xn = xi < a ? xi : a;
                }
            }
            double yn = yl + (yi - yl) * (xn - xl) / (xi - xl);
            /* the first THERMR point at or above xn (aceth.f90:432-443); xn only grows */
            while (k < nep - 1 && x[k] < xn) k++;
            double lo = x[k - 1], hi = x[k];
            const double *ck = cos + k * nang, *cl = cos + (k - 1) * nang;
            if (fabs(xn - hi) < eps) memcpy(c, ck, nang * sizeof(double));
            else if (fabs(xn - lo) < eps) memcpy(c, cl, nang * sizeof(double));
            else for (int j = 0; j < nang; j++) c[j] = cl[j] + (ck[j] - cl[j]) * (xn - lo) / (hi - lo);
            if (!ace_push(o, xn, yn, fract, c)) { ok = 0; goto done; }
            xl = xn; yl = yn; fract = 1.0; total = 0.0;
            if (xl < xi) continue;
            break;
        }
        xl = xi; yl = yi;
    }
 done:
    free(x); free(y); free(cos); free(c);
    return ok;
}

static PyObject *py_ace_energy(PyObject *self, PyObject *args) {
    PyObject *ox, *oy;
    double xsi;
    int nang;
    Py_buffer bx = {0}, by = {0};
    if (!PyArg_ParseTuple(args, "OOdi", &ox, &oy, &xsi, &nang)) return NULL;
    if (!buffer(ox, &bx, 0, 0)) return NULL;
    Py_ssize_t nep = bx.len / 8;
    if (!buffer(oy, &by, 0, 0)) { PyBuffer_Release(&bx); return NULL; }
    int nl = nep ? (int)(by.len / 8 / nep) : 0;
    if (nang < 1 || (nep && (by.len != nep * nl * 8 || nl < nang + 1))) {
        PyBuffer_Release(&bx); PyBuffer_Release(&by);
        PyErr_SetString(PyExc_ValueError, "ace_energy: y must be (len(x), >= nang + 1)");
        return NULL;
    }
    AceOut o = {NULL, NULL, NULL, NULL, 0, 0, nang};
    int ok;
    Py_BEGIN_ALLOW_THREADS
    ok = ace_energy(bx.buf, by.buf, nep, nl, xsi, nang, &o);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&bx); PyBuffer_Release(&by);
    PyObject *result = NULL;
    if (!ok) PyErr_NoMemory();
    else result = Py_BuildValue("y#y#y#y#", (char *)o.x, o.n * 8, (char *)o.y, o.n * 8,
                                (char *)o.f, o.n * 8, (char *)o.c, o.n * nang * 8);
    free(o.x); free(o.y); free(o.f); free(o.c);
    return result;
}

static PyMethodDef methods[] = {
    {"ace_energy", py_ace_energy, METH_VARARGS,
     "THERMR tpend then ACER acesix iwt=2 on one incident energy, bit for bit the Python reference."},
    {"endf11", py_endf11, METH_VARARGS, "Round an array in place as NJOY writes ENDF fields (a11)."},
    {"calcem", calcem, METH_VARARGS, "THERMR calcem for one incident energy: (count, xsi)."},
    {"sig", py_sig, METH_VARARGS, "THERMR sig: the double-differential cross section."},
    {"sigfig", py_sigfig, METH_VARARGS, "NJOY sigfig."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_thermal_native", NULL, 0, methods};
PyMODINIT_FUNC PyInit__thermal_native(void) {
#ifdef _WIN32
    HMODULE ucrt = GetModuleHandleW(L"ucrtbase.dll");
    if (!ucrt) ucrt = LoadLibraryW(L"ucrtbase.dll");
    FARPROC found = ucrt ? GetProcAddress(ucrt, "pow") : NULL;
    if (!found) {
        PyErr_SetString(PyExc_ImportError, "native thermal: the Universal CRT's pow is not available");
        return NULL;
    }
    crt_pow = (pow_function)(void (*)(void))found;
#endif
    fill_p10();
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "API_VERSION", 3) < 0) { Py_DECREF(m); return NULL; }
    return m;
}
