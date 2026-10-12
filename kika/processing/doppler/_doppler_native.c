/* Analytic Doppler panel integration, strict float64, no NumPy C ABI.
 * Conditioned closed Gaussian moments with local Taylor fallback.
 * No fast math, thinning, changed support or sampled integration.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Doppler requires strict floating point"
#endif
typedef char doppler_requires_float64[(sizeof(double)==8 && DBL_MANT_DIG==53)?1:-1];
typedef struct { double sum, correction; } Sum;
static void add(Sum *a, double v) {
    double t=a->sum+v;
    a->correction += fabs(a->sum)>=fabs(v) ? (a->sum-t)+v : (v-t)+a->sum;
    a->sum=t;
}
static double total(Sum a) { return a.sum+a.correction; }
static double weights[80][5];

/* The kernel's own exp and erfc. The MinGW-w64 UCRT libm takes 35 ns per exp
 * and 83 ns per erfc, and they are most of the cost of a panel; these take a
 * few. Accuracy is checked against the reference values in
 * tests/test_native.py (special): <= 2 ulp for kexp, <= 4e-16 relative for
 * kerfc, inside the 16-64 eps provisions the integrator charges.
 *
 * kexp: Cody-Waite reduction x = k ln2 + r, |r| <= ln2/2, with fdlibm's
 * split of ln2, and the degree-13 Taylor polynomial of e^r (truncation
 * < 4e-18 relative), scaled by 2^k through the exponent bits. */
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

/* e^-(y^2) for y >= 0 without the rounding of y*y: y^2 = hi + lo exactly
 * (Dekker's product, Veltkamp split), and e^-y^2 = e^-hi (1 - lo), whose
 * neglected term lo^2/2 is below 4e-27 for y <= 27.3. */
static double kgauss(double y) {
    if(y>27.3) return 0.;
    double hi=y*y,c=134217729.*y,yh=c-(c-y),yl=y-yh;
    double lo=((yh*yh-hi)+2*yh*yl)+yl*yl;
    return kexp(-hi)*(1-lo);
}

/* erfc(y), y >= 0, by W. J. Cody's rational Chebyshev approximations
 * (Math. Comp. 23 (1969) 631; SPECFUN CALERF): relative error of the
 * approximations below 1e-18, plus the rounding of their evaluation.
 * Returns e^-(y^2) through *gauss as well, which the caller shares. */
static double kerfc(double y,double *gauss) {
    static const double A[5]={3.16112374387056560e00,1.13864154151050156e02,3.77485237685302021e02,
                              3.20937758913846947e03,1.85777706184603153e-1};
    static const double B[4]={2.36012909523441209e01,2.44024637934444173e02,1.28261652607737228e03,
                              2.84423683343917062e03};
    static const double C[9]={5.64188496988670089e-1,8.88314979438837594e00,6.61191906371416295e01,
                              2.98635138197400131e02,8.81952221241769090e02,1.71204761263407058e03,
                              2.05107837782607147e03,1.23033935479799725e03,2.15311535474403846e-8};
    static const double D[8]={1.57449261107098347e01,1.17693950891312499e02,5.37181101862009858e02,
                              1.62138957456669019e03,3.29079923573345963e03,4.36261909014324716e03,
                              3.43936767414372164e03,1.23033935480374942e03};
    static const double P[6]={3.05326634961232344e-1,3.60344899949804439e-1,1.25781726111229246e-1,
                              1.60837851487422766e-2,6.58749161529837803e-4,1.63153871373020978e-2};
    static const double Q[5]={2.56852019228982242e00,1.87295284992346725e00,5.27905102951428412e-1,
                              6.05183413124413191e-2,2.33520497626869185e-3};
    double g=kgauss(y);*gauss=g;
    if(y<=.46875) {
        double ysq=y*y,xnum=A[4]*ysq,xden=ysq;
        for(int i=0;i<3;i++) {xnum=(xnum+A[i])*ysq;xden=(xden+B[i])*ysq;}
        return 1.-y*(xnum+A[3])/(xden+B[3]);
    }
    if(y<=4.) {
        double xnum=C[8]*y,xden=y;
        for(int i=0;i<7;i++) {xnum=(xnum+C[i])*y;xden=(xden+D[i])*y;}
        return g*(xnum+C[7])/(xden+D[7]);
    }
    if(g==0) return 0.;
    double ysq=1./(y*y),xnum=P[5]*ysq,xden=ysq;
    for(int i=0;i<4;i++) {xnum=(xnum+P[i])*ysq;xden=(xden+Q[i])*ysq;}
    double result=ysq*(xnum+P[4])/(xden+Q[4]);
    return g*(5.6418958354775628695e-1-result)/y;
}

/* For |c_k| <= A/k |c_(k-1)| + B/k |c_(k-2)|, the coefficient
 * tail after n is bounded by
 * ((A+B)|c_n| + B|c_(n-1)|)/(n+1-A-B).
 * Sum the recurrence for k>n and replace 1/k by 1/(n+1).
 * Gaussian moment denominators further divide this by n+2.
 * Coupled low-z coefficients use |D|+|H| and the row-sum A.
 * Padding protects rounding of this diagnostic; it is not interval arithmetic.
 */
static double coefficient_tail(double A,double B,double cn,double prior,Py_ssize_t n) {
    if(cn==0 && prior==0) return 0;
    double denominator=n+1-A-B;
    if(denominator<=0) return INFINITY;
    return ((A+B)*cn+B*prior)/denominator*(1+128*DBL_EPSILON);
}

static Py_ssize_t moments(double s,double d,double z,Py_ssize_t terms,
                    double *dif,double *other,double *out,double *rem,double *norm,int nmom,const double *poly) {
    /* exp(-x) is exactly 0 in double for x > 745.14: skipping the call for
     * x > 800 gives the same bits. */
    double plus=kgauss(fabs(s)),t=z+s,minus=4*z*t>800?0.:plus*kexp(-4*z*t);
    Py_ssize_t used=terms;
    double omitted=0;
    Sum seen={0};
    if (z<.125) {
        dif[0]=plus*(-expm1(-4*z*t)); other[0]=plus+minus;
        add(&seen,fabs(dif[0]));
        for (Py_ssize_t n=1;n<terms;n++) {
            double a=-2*t*d*dif[n-1]+2*z*d*other[n-1];
            double b=-2*t*d*other[n-1]+2*z*d*dif[n-1];
            if (n>1) { a-=2*d*d*dif[n-2]; b-=2*d*d*other[n-2]; }
            dif[n]=a/n; other[n]=b/n;
            add(&seen,fabs(dif[n]));
            if(terms==80 && n>=(poly==NULL?8:2)) {
                double bound=coefficient_tail(2*(fabs(t)+z)*d,2*d*d,
                    fabs(dif[n])+fabs(other[n]),fabs(dif[n-1])+fabs(other[n-1]),n);
                if(bound/(n+2)<=4*DBL_EPSILON*total(seen)) {used=n+1;omitted=bound;break;}
            }
        }
    } else if (minus==0) {
        /* The reflected Gaussian has underflowed (every point but the lowest
         * energies): its coefficients are all exact zeros, contribute 0 to
         * the tail bound and to the difference, so the loop below without
         * them gives the same bits at half the work. */
        dif[0]=plus;
        add(&seen,fabs(plus));
        for (Py_ssize_t n=1;n<terms;n++) {
            double a=-2*s*d*dif[n-1];
            if (n>1) a-=2*d*d*dif[n-2];
            dif[n]=a/n;
            add(&seen,fabs(dif[n]));
            if(terms==80 && n>=(poly==NULL?8:2)) {
                double bound=coefficient_tail(2*fabs(s)*d,2*d*d,fabs(dif[n]),fabs(dif[n-1]),n);
                if(bound/(n+2)<=4*DBL_EPSILON*total(seen)) {used=n+1;omitted=bound;break;}
            }
        }
    } else {
        dif[0]=plus; other[0]=minus;
        add(&seen,fabs(plus-minus));
        for (Py_ssize_t n=1;n<terms;n++) {
            double a=-2*s*d*dif[n-1],b=-2*(s+2*z)*d*other[n-1];
            if (n>1) { a-=2*d*d*dif[n-2]; b-=2*d*d*other[n-2]; }
            dif[n]=a/n; other[n]=b/n;
            add(&seen,fabs(dif[n]-other[n]));
            if(terms==80 && n>=(poly==NULL?8:2)) {
                double bound=coefficient_tail(2*fabs(s)*d,2*d*d,fabs(dif[n]),fabs(dif[n-1]),n)
                    +coefficient_tail(2*fabs(s+2*z)*d,2*d*d,fabs(other[n]),fabs(other[n-1]),n);
                if(bound/(n+2)<=4*DBL_EPSILON*total(seen)) {used=n+1;omitted=bound;break;}
            }
        }
        for (Py_ssize_t n=0;n<used;n++) dif[n]-=other[n];
    }
    Sum magnitude={0},tail={0};
    for (Py_ssize_t n=0;n<used;n++) {
        add(&magnitude,fabs(dif[n]));
        if(n>=used-4) add(&tail,fabs(dif[n]));
    }
    *norm=total(magnitude)+omitted;
    *rem=used<terms?omitted/(used+1):4*total(tail);
    /* Keep the reference's default rounding provision. Compensated
     * reductions and rounded divisions also fit within this provision. */
    if (terms==80) *rem+=83*DBL_EPSILON*(*norm);
    if(poly!=NULL) {
        /* Contract the polynomial into each coefficient once. The existing
         * 256*EPS*norm*|poly| provision covers these <=5 rounded products
         * and additions; compensated accumulation retains signed sources. */
        Sum a={0};
        for(Py_ssize_t n=0;n<used;n++) {
            double weight=0;
            for(int j=0;j<nmom;j++) weight+=poly[j]*(terms==80?weights[n][j]:1./(n+j+1));
            add(&a,dif[n]*weight);
        }
        out[0]=total(a);
    } else {
        for (int j=0;j<nmom;j++) {
            Sum a={0};
            for (Py_ssize_t n=0;n<used;n++) add(&a,terms==80?dif[n]*weights[n][j]:dif[n]/(n+j+1));
            out[j]=total(a);
        }
    }
    return used;
}

static PyObject *moment_probe(PyObject *self,PyObject *args) {
    double s,d,z;Py_ssize_t terms;
    if(!PyArg_ParseTuple(args,"dddn",&s,&d,&z,&terms)) return NULL;
    if(!isfinite(s) || !isfinite(d) || !isfinite(z) || z<=0 ||
       d<=0 || d>.25 || s < -z || fabs(s)>26 || terms<8 || terms>200000) {
        PyErr_SetString(PyExc_ValueError,"invalid native Doppler moments");return NULL;
    }
    double *work=PyMem_Malloc(2*terms*sizeof(double));if(!work)return PyErr_NoMemory();
    double out[5],rem,norm;
    Py_ssize_t used=moments(s,d,z,terms,work,work+terms,out,&rem,&norm,5,NULL);
    PyMem_Free(work);
    return Py_BuildValue("(ddddd)ddn",out[0],out[1],out[2],out[3],out[4],rem,norm,used);
}

/* Gaussian moments in the source panel's unit coordinate. Differences use
 * erfc on same-sign tails. Absolute roundoff estimates propagate through the
 * recurrence and final polynomial contraction; ill-conditioned results fall
 * back to the local series. This is not interval arithmetic. */
static void single_gaussian(double a,double d,double *m,double *e,int n) {
    const double eps=DBL_EPSILON,rootpi=1.772453850905516027298167483341145;
    if(a>40) {
        for(int j=0;j<n;j++) {m[j]=0;e[j]=DBL_MIN;}
        return;
    }
    double b=a+d,ga=exp(-a*a),gb=exp(-b*b),fa,fb,difference;
    if(a>=0) {fa=erfc(a);fb=erfc(b);difference=fa-fb;}
    else if(b<=0) {fa=erfc(-b);fb=erfc(-a);difference=fa-fb;}
    else {fa=erf(b);fb=erf(a);difference=fa-fb;}
    m[0]=rootpi/(2*d)*difference;
    e[0]=16*eps*rootpi/(2*d)*(fabs(fa)+fabs(fb))+DBL_MIN;
    if(n==1) return;
    double inv=1/(2*d*d),ad=a/d;
    double ea=16*eps*(1+a*a)*ga+DBL_MIN,eb=16*eps*(1+b*b)*gb+DBL_MIN;
    double first=(ga-gb)*inv,second=ad*m[0];
    m[1]=first-second;
    e[1]=inv*(ea+eb)+fabs(ad)*e[0]+32*eps*(fabs(first)+fabs(second))+DBL_MIN;
    for(int j=2;j<n;j++) {
        double u=(j-1)*inv*m[j-2],v=ad*m[j-1],w=gb*inv;
        m[j]=u-v-w;
        e[j]=(j-1)*inv*e[j-2]+fabs(ad)*e[j-1]+inv*eb
             +32*eps*(fabs(u)+fabs(v)+fabs(w))+DBL_MIN;
    }
}

static double gaussian_max(double a,double b) {
    double near=a<=0 && b>=0?0:fmin(fabs(a),fabs(b));
    return exp(-near*near)*(1+8*DBL_EPSILON)+DBL_MIN;
}

static int closed_panel(double s,double d,double z,const double *poly,int n,
                        double *value,double *error,double *norm) {
    /* Avoid building a recurrence which predictably loses significance.
     * The final acceptance depends on the contracted error, not this width. */
    if(d<.05 || d>52 || z<.125) return 0;
    double plus[5],minus[5],ep[5],em[5];
    single_gaussian(s,d,plus,ep,n);
    single_gaussian(s+2*z,d,minus,em,n);
    Sum sum={0},uncertainty={0},absolute={0};
    for(int j=0;j<n;j++) {
        double contribution=poly[j]*(plus[j]-minus[j]);
        add(&sum,contribution);add(&absolute,fabs(contribution));
        add(&uncertainty,fabs(poly[j])*(ep[j]+em[j]+8*DBL_EPSILON*(fabs(plus[j])+fabs(minus[j]))));
    }
    *value=total(sum);
    *error=total(uncertainty)+32*DBL_EPSILON*total(absolute);
    *norm=gaussian_max(s,s+d)+gaussian_max(s+2*z,s+d+2*z);
    return isfinite(*value) && isfinite(*error) && isfinite(*norm)
           && *error<=3e-13*fabs(*value);
}

static PyObject *closed_probe(PyObject *self,PyObject *args) {
    double s,d,z;PyObject *coefficients;
    if(!PyArg_ParseTuple(args,"dddO",&s,&d,&z,&coefficients)) return NULL;
    if(!isfinite(s) || !isfinite(d) || !isfinite(z) || d<=0 || d>52 || z<=0 ||
       s < -z || fabs(s)>26 || !PyTuple_Check(coefficients) ||
       PyTuple_GET_SIZE(coefficients)<1 || PyTuple_GET_SIZE(coefficients)>5) {
        PyErr_SetString(PyExc_ValueError,"invalid closed Doppler moments");return NULL;
    }
    double poly[5]={0},value,error,norm;int n=(int)PyTuple_GET_SIZE(coefficients);
    for(int j=0;j<n;j++) {
        poly[j]=PyFloat_AsDouble(PyTuple_GET_ITEM(coefficients,j));
        if(PyErr_Occurred())return NULL;
        if(!isfinite(poly[j])) {PyErr_SetString(PyExc_ValueError,"invalid polynomial");return NULL;}
    }
    if(!closed_panel(s,d,z,poly,n,&value,&error,&norm)) Py_RETURN_NONE;
    return Py_BuildValue("ddd",value,error,norm);
}

/* Prepared immutable numeric panels own the original tuple (and its labels).
 * They avoid Python conversions and sqrt/slope preparation at each point. */
typedef struct {
    double lo,hi,v0,v1,anchor,ylo,yhi,slope,dsh;
    int lin,inv,constant;
    PyObject *owner;
} NumericPanel;
typedef struct {Py_ssize_t size;PyObject *source;NumericPanel *data;} PackedPanels;
static const char *capsule_name="kika.doppler.panels.v1";

static int read_panel(PyObject *panel,NumericPanel *v) {
    if(!PyTuple_Check(panel) || PyTuple_GET_SIZE(panel)!=7) {
        PyErr_SetString(PyExc_ValueError,"invalid native Doppler panel");return 0;
    }
    v->lo=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,0));v->hi=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,1));
    v->v0=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,2));v->v1=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,3));
    v->anchor=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,6));
    PyObject *kind=PyTuple_GET_ITEM(panel,4);v->owner=PyTuple_GET_ITEM(panel,5);
    if(PyErr_Occurred())return 0;
    v->lin=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"linlin")==0;
    v->inv=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"one_over_v")==0;
    v->constant=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"constant")==0;
    if(!isfinite(v->lo) || v->lo<0 || !(v->hi>v->lo) || !isfinite(v->v0) || !isfinite(v->v1) ||
       (!v->lin && !v->inv && !v->constant) || (v->lin && !isfinite(v->hi)) ||
       (v->inv && (!isfinite(v->anchor) || v->anchor<=0)) || !PyUnicode_Check(v->owner)) {
        PyErr_SetString(PyExc_ValueError,"invalid native Doppler panel");return 0;
    }
    v->ylo=sqrt(v->lo);v->yhi=sqrt(v->hi);
    v->slope=v->lin?(v->v1-v->v0)/(v->hi-v->lo):0;
    /* The panel's width in velocity, independent of the point: ds = dsh/h. */
    v->dsh=isinf(v->hi)?INFINITY:(v->hi-v->lo)/(v->yhi+v->ylo);
    return 1;
}

static void free_packed(PyObject *capsule) {
    PackedPanels *p=PyCapsule_GetPointer(capsule,capsule_name);
    if(p==NULL) {PyErr_Clear();return;}
    Py_XDECREF(p->source);PyMem_Free(p->data);PyMem_Free(p);
}

static PyObject *prepare_panels(PyObject *self,PyObject *source) {
    if(!PyTuple_Check(source)) {PyErr_SetString(PyExc_ValueError,"expected tuple panels");return NULL;}
    Py_ssize_t n=PyTuple_GET_SIZE(source);
    if(n>(PY_SSIZE_T_MAX-(Py_ssize_t)sizeof(PackedPanels))/(Py_ssize_t)sizeof(NumericPanel)) return PyErr_NoMemory();
    PackedPanels *p=PyMem_Calloc(1,sizeof(PackedPanels));if(p==NULL)return PyErr_NoMemory();
    p->data=PyMem_Calloc(n?n:1,sizeof(NumericPanel));
    if(p->data==NULL) {PyMem_Free(p);return PyErr_NoMemory();}
    p->size=n;
    for(Py_ssize_t i=0;i<n;i++) {
        if(!read_panel(PyTuple_GET_ITEM(source,i),&p->data[i])) {PyMem_Free(p->data);PyMem_Free(p);return NULL;}
        if(i>0 && (p->data[i].lo<p->data[i-1].lo || p->data[i].hi<p->data[i-1].hi)) {
            PyErr_SetString(PyExc_ValueError,"unordered native panels");PyMem_Free(p->data);PyMem_Free(p);return NULL;
        }
    }
    Py_INCREF(source);p->source=source;
    PyObject *capsule=PyCapsule_New(p,capsule_name,free_packed);
    if(capsule==NULL) {Py_DECREF(source);PyMem_Free(p->data);PyMem_Free(p);return NULL;}
    return Py_BuildValue("Nn",capsule,(Py_ssize_t)sizeof(PackedPanels)+(n?n:1)*(Py_ssize_t)sizeof(NumericPanel));
}

/* Node-shared Gaussian moments (the BROADR/SIGMA1 arrangement): the upper
 * tails U_k(w) = int_w^inf t^k e^-t^2 dt, w >= 0, k <= 4, from
 * U_0 = sqrt(pi)/2 erfc(w), U_1 = e^-w^2/2 and
 * U_k = (k-1)/2 U_(k-2) + w^(k-1) e^-w^2/2 -- every term positive, so each
 * U_k is good to a few ulps. A node is shared by the two panels that meet
 * there, so it is computed once. */
typedef struct {double s;double u[5];int valid;} NodeMoments;
static const double U_ZERO[5]={0.886226925452758013649083741670572591,0.5,
    0.443113462726379006824541870835286296,0.5,0.664670194089568510236812806252929444};

static const double *node_moments(NodeMoments *cache,int slot,double s) {
    NodeMoments *c=&cache[slot];
    if(c->valid && c->s==s) return c->u;
    double w=fabs(s);
    if(isinf(w)) {for(int k=0;k<5;k++)c->u[k]=0;}
    else {
        double gauss,complement=kerfc(w,&gauss),g=gauss/2;
        c->u[0]=0.886226925452758013649083741670572591*complement;
        c->u[1]=g;
        c->u[2]=.5*c->u[0]+w*g;
        c->u[3]=c->u[1]+w*w*g;
        c->u[4]=1.5*c->u[2]+w*w*w*g;
    }
    c->s=s;c->valid=1;
    return c->u;
}

/* int_a^b t^k e^-t^2 dt for k < n, and a bound on the magnitude of the terms
 * combined (for the rounding provision). L_k(s) = (-1)^k U_k(-s) for s <= 0. */
static void node_integral(NodeMoments *cache,double a,double b,int n,double *I,double *mag,
                          double *ga,double *gb) {
    const double *ua=node_moments(cache,0,a);
    double ub[5];
    const double *pb=node_moments(cache,1,b);
    for(int k=0;k<5;k++) ub[k]=pb[k];
    *ga=2*ua[1];*gb=2*ub[1];               /* e^-a^2, e^-b^2 */
    for(int k=0;k<n;k++) {
        double sign=(k&1)?-1.:1.;
        if(b<=0) {I[k]=sign*(ub[k]-ua[k]);mag[k]=ub[k]+ua[k];}
        else if(a>=0) {I[k]=ua[k]-ub[k];mag[k]=ua[k]+ub[k];}
        else {I[k]=sign*(U_ZERO[k]-ua[k])+(U_ZERO[k]-ub[k]);mag[k]=2*U_ZERO[k]+ua[k]+ub[k];}
    }
    /* The right node of this panel is the left node of the next. */
    NodeMoments t=cache[0];cache[0]=cache[1];cache[1]=t;
}

static int store_component(PyObject *components,PyObject *owner,double value) {
    PyObject *number=PyFloat_FromDouble(value);if(number==NULL)return -1;
    int rc=PyDict_SetItem(components,owner,number);Py_DECREF(number);return rc;
}

static PyObject *integrate(PyObject *self,PyObject *args) {
    PyObject *panels,*components=NULL,*result=NULL;
    Py_ssize_t first,last,terms,limit,count=0,reserved=0;int hybrid=1;
    double energy,h,lower,upper;
    double *work=NULL;
    if(!PyArg_ParseTuple(args,"Onnddddnn|p",&panels,&first,&last,&energy,&h,&lower,&upper,&terms,&limit,&hybrid)) return NULL;
    PackedPanels *packed=PyCapsule_IsValid(panels,capsule_name)?PyCapsule_GetPointer(panels,capsule_name):NULL;
    Py_ssize_t size=packed?packed->size:(PyTuple_Check(panels)?PyTuple_GET_SIZE(panels):-1);
    if(size<0 || first<0 || last<first || last>size ||
       terms<8 || terms>PY_SSIZE_T_MAX/(2*(Py_ssize_t)sizeof(double)) || limit<1 ||
       !isfinite(energy) || energy<=0 || !isfinite(h) || h<=0 ||
       !isfinite(lower) || !isfinite(upper) || lower>=upper) {
        PyErr_SetString(PyExc_ValueError,"invalid native Doppler arguments"); return NULL;
    }
    double x=sqrt(energy),z=x/h;
    if(!isfinite(z*z) || z*z==0) { PyErr_SetString(PyExc_ArithmeticError,"panel overflow");return NULL; }
    work=PyMem_Malloc(2*terms*sizeof(double));
    if(work==NULL) return PyErr_NoMemory();
    components=PyDict_New(); if(components==NULL) goto done;
    PyObject *component_owner=NULL; double component_value=0;
    /* Node-shared closed moments wherever the reflected Gaussian
     * e^-(s+2z)^2 is exactly 0 in double over the whole window (s >= lower,
     * so s+2z >= 2z+lower > 27.3 and e^-745 underflows): the series takes
     * the same zero there. */
    NodeMoments node_cache[2]={{0},{0}};
    int node_path=hybrid && terms==80 && 2*z+lower>27.3;
    /* The value is summed compensated (panels of either sign can cancel);
     * the error bounds and |values| are sums of nonnegative terms, whose
     * rounding the final (1 + 2 n eps) factor covers. */
    Sum values={0};double errors=0,absolute=0;
    /* A panel's right node is the next panel's left: its s is reused. */
    double previous_hi=NAN,previous_s=0,invz=1/z;
    for(Py_ssize_t p=first;p<last;p++) {
        NumericPanel temporary;
        const NumericPanel *v;
        if(packed) v=&packed->data[p];
        else {
            if(!read_panel(PyTuple_GET_ITEM(panels,p),&temporary))goto done;
            v=&temporary;
        }
        double lo=v->lo,hi=v->hi,v0=v->v0,v1=v->v1,anchor=v->anchor;
        int lin=v->lin,inv=v->inv,constant=v->constant;
        PyObject *owner=v->owner;
        int include=0;
        Sum contribution={0};
        if(v0==0 && v1==0) include=1;
        else {
            double ylo=v->ylo,yhi=v->yhi;
            double slo=lo==previous_hi?previous_s:(lo-energy)/(h*(ylo+x));
            double shi=isinf(hi)?INFINITY:(hi-energy)/(h*(yhi+x));
            previous_hi=hi;previous_s=shi;
            double left=fmax(lower,slo),right=fmin(upper,shi);
            if(left>=right) continue;
            double ds=v->dsh/h;
            int narrow=ds<1;
            if(ds<=0) {PyErr_SetString(PyExc_ArithmeticError,"panel overflow");goto done;}
            double qleft=narrow?fmax(0,(lower-slo)/ds):0;
            double qright=narrow?fmin(1,(upper-slo)/ds):1;
            double width=narrow?(qright-qleft)*ds:right-left;
            double subdivisions=fmax(1,ceil(width/.25));
            if(!isfinite(subdivisions) || subdivisions>limit-reserved) {
                PyErr_SetString(PyExc_OverflowError,"analytic subpanel limit");goto done;
            }
            Py_ssize_t planned=(Py_ssize_t)subdivisions;reserved+=planned;
            /* A panel at least 1/1000 of a Doppler width wide is integrated in
             * one piece from its two nodes' moments: the difference of the
             * node values loses at most eps/width of the panel's part, which
             * the provision below charges. Narrower panels (an isolated
             * spike, or a node-dense flank) keep the conditioned series. */
            if(node_path && width>=1e-3) {
                double a=left,b=right,c[5]={0},cs[5]={0},S[3],Sabs[3];int nc;
                double iz=invz;
                if(inv) {
                    double w=v0*sqrt(anchor)/x;
                    c[0]=w;c[1]=w*iz;cs[0]=fabs(w);cs[1]=fabs(w)*iz;nc=2;
                } else {
                    if(lin) {
                        double slope=v->slope;
                        S[0]=v0+(v1-v0)*((energy-lo)/(hi-lo));S[1]=slope*2*x*h;S[2]=slope*h*h;
                        Sabs[0]=fabs(v0)+fabs(v1-v0)*(1+fabs(energy-lo)/(hi-lo));
                        Sabs[1]=fabs(S[1]);Sabs[2]=fabs(S[2]);
                    } else {S[0]=v0;S[1]=0;S[2]=0;Sabs[0]=fabs(v0);Sabs[1]=0;Sabs[2]=0;}
                    /* P(s) = sigma(s) (1+s/z)^2, sigma(s) = S0+S1 s+S2 s^2. */
                    c[0]=S[0];
                    c[1]=S[1]+2*S[0]*iz;
                    c[2]=S[2]+2*S[1]*iz+S[0]*iz*iz;
                    c[3]=2*S[2]*iz+S[1]*iz*iz;
                    c[4]=S[2]*iz*iz;
                    cs[0]=Sabs[0];
                    cs[1]=Sabs[1]+2*Sabs[0]*iz;
                    cs[2]=Sabs[2]+2*Sabs[1]*iz+Sabs[0]*iz*iz;
                    cs[3]=2*Sabs[2]*iz+Sabs[1]*iz*iz;
                    cs[4]=Sabs[2]*iz*iz;
                    nc=lin?5:3;
                }
                double I[5],mag[5],ea,eb;
                node_integral(node_cache,a,b,nc,I,mag,&ea,&eb);
                /* Plain sum: <= nc eps sum|c_k I_k|, inside the 16 eps provision. */
                double sum=0,bound=0;
                for(int k=0;k<nc;k++) {
                    sum+=c[k]*I[k];
                    /* Coefficient rounding (64 eps of their scale) plus the
                     * node values' and the difference's rounding. */
                    bound+=64*DBL_EPSILON*cs[k]*mag[k]+16*DBL_EPSILON*fabs(c[k])*mag[k];
                }
                const double rootpi=1.772453850905516027298167483341145;
                double value=sum/rootpi;
                /* Float64 position of the two nodes, |dP e^-s^2/ds| times it. */
                double pa=0,pb=0;
                for(int k=nc-1;k>=0;k--) {pa=pa*fabs(a)+cs[k];pb=isinf(b)?0:pb*fabs(b)+cs[k];}
                double position=64*DBL_EPSILON*((1+fabs(a))*(1+fabs(a))*pa*ea+(isinf(b)?0:(1+fabs(b))*(1+fabs(b))*pb*eb));
                double err=(bound+position)/rootpi;
                if(!isfinite(value) || !isfinite(err)) {PyErr_SetString(PyExc_ArithmeticError,"panel overflow");goto done;}
                add(&values,value);errors+=err;absolute+=fabs(value);add(&contribution,value);
                count++;include=1;
                goto panel_done;
            }
            Py_ssize_t nsub=hybrid && terms==80 && width>=1?1:planned;
            for(Py_ssize_t i=0;i<nsub;i++) {
                double d=width/nsub;
                double s=narrow?slo+(qleft+i*(qright-qleft)/nsub)*ds:left+i*d;
                double yr=1+s/z,dr=d/z,poly[5]={0},sig[3]={0}; int ncoeff;
                if(inv) {
                    poly[0]=v0*sqrt(anchor)/x*yr;poly[1]=v0*sqrt(anchor)/x*dr;ncoeff=2;
                } else {
                    int nsig;
                    if(lin) {
                        double sig0,slope=v->slope;
                        if(narrow) {
                            double q=qleft+i*(qright-qleft)/nsub,dy=ds*h;
                            double de0=q*dy*(2*ylo+q*dy);
                            sig0=v0+(v1-v0)*(de0/(hi-lo));
                        } else {
                            double de0=(energy-lo)+h*s*(2*x+h*s);
                            sig0=v0+(v1-v0)*(de0/(hi-lo));
                        }
                        sig[0]=sig0;sig[1]=slope*(2*(x+h*s)*h*d);sig[2]=slope*((h*d)*(h*d));nsig=3;
                    } else {sig[0]=v0;nsig=1;}
                    double quadratic[3]={yr*yr,2*yr*dr,dr*dr};
                    for(int a=0;a<3;a++) for(int b=0;b<nsig;b++) poly[a+b]+=quadratic[a]*sig[b];
                    ncoeff=nsig+2;
                }
                double mom[5],rem,norm;
                double closed_error=0;
                int closed=hybrid && terms==80 && width>=1 && nsub==1 && closed_panel(s,d,z,poly,ncoeff,&mom[0],&closed_error,&norm);
                if(!closed && nsub==1 && planned>1) {nsub=planned;i=-1;continue;}
                if(!closed) moments(s,d,z,terms,work,work+terms,mom,&rem,&norm,ncoeff,poly);
                Sum pnorm={0};
                for(int j=0;j<ncoeff;j++) add(&pnorm,fabs(poly[j]));
                double factor=d/sqrt(3.141592653589793238462643383279502884);
                double value=factor*mom[0],err=closed?factor*closed_error:factor*total(pnorm)*(rem+256*DBL_EPSILON*norm);
                /* Float64 geometry sensitivity, including partially clipped
                 * narrow-panel width and interpolated height. */
                double relative=64*DBL_EPSILON*(1+fabs(s)+d)*(1+fabs(s)+d),sigma_error=0;
                int left_touch=narrow && lower>=slo-32*DBL_EPSILON*(fabs(slo)+fabs(lower)+ds);
                int right_touch=narrow && upper<=slo+ds+32*DBL_EPSILON*(fabs(slo)+fabs(upper)+ds);
                if(narrow && (left_touch || right_touch)) {
                    double boundary=(left_touch?fabs(lower):0)+(right_touch?fabs(upper):0);
                    double displacement=32*DBL_EPSILON*(fabs(slo)+boundary+ds);
                    relative+=displacement/d;
                    if(lin) sigma_error+=fabs(v1-v0)*displacement/ds;
                }
                if(lin) {
                    double scale=fabs(v0)+fabs(v1-v0);
                    if(!narrow) scale+=fabs(v1-v0)*(fabs(energy-lo)+fabs(h*s)*(2*x+fabs(h*s)))/(hi-lo);
                    sigma_error+=64*DBL_EPSILON*scale;
                } else if(constant) sigma_error+=64*DBL_EPSILON*fabs(v0);
                err+=factor*norm*(total(pnorm)*relative+sigma_error*(fabs(yr)+dr)*(fabs(yr)+dr));
                if(!isfinite(value) || !isfinite(err)) {PyErr_SetString(PyExc_ArithmeticError,"panel overflow");goto done;}
                add(&values,value);errors+=err;absolute+=fabs(value);add(&contribution,value);
                count++;include=1;
            }
        }
        panel_done:
        if(include) {
            /* Preserve the original ordered additions, including revisited
             * labels, but materialise one Python float per owner run. */
            if(component_owner==NULL || !PyUnicode_CheckExact(owner) || !PyUnicode_CheckExact(component_owner)
                || (owner!=component_owner && PyUnicode_Compare(owner,component_owner)!=0)) {
                if(component_owner!=NULL && store_component(components,component_owner,component_value)<0)goto done;
                PyObject *previous=PyDict_GetItemWithError(components,owner);
                if(previous==NULL && PyErr_Occurred())goto done;
                component_value=previous?PyFloat_AsDouble(previous):0;
                component_owner=owner;
            }
            component_value+=total(contribution);
        }
        if((p-first)%1024==0 && PyErr_CheckSignals()<0) goto done;
    }
    if(component_owner!=NULL && store_component(components,component_owner,component_value)<0)goto done;
    double inflation=1+2*(double)(count+1)*DBL_EPSILON;
    double value=total(values),error=errors*inflation+8*DBL_EPSILON*absolute*inflation;
    if(!isfinite(value) || !isfinite(error)) {
        PyErr_SetString(PyExc_ArithmeticError,"aggregate overflow");goto done;
    }
    result=Py_BuildValue("ddOn",value,error,components,count);
done:
    Py_XDECREF(components);PyMem_Free(work);return result;
}
/* Scalar point control in a bounded block. The shared integrator retains its
 * series/closed conditioning, compensated sums and resource preflight. */
static void point_error(double energy,const char *code,double budget) {
    PyObject *type,*value,*trace;
    PyErr_Fetch(&type,&value,&trace);PyErr_NormalizeException(&type,&value,&trace);
    if(value!=NULL) {
        PyObject *e=PyFloat_FromDouble(energy),*c=PyUnicode_FromString(code),*b=PyFloat_FromDouble(budget);
        if(e!=NULL && PyObject_SetAttrString(value,"energy",e)<0)PyErr_Clear();
        if(c!=NULL && PyObject_SetAttrString(value,"doppler_code",c)<0)PyErr_Clear();
        if(b!=NULL && PyObject_SetAttrString(value,"budget",b)<0)PyErr_Clear();
        Py_XDECREF(e);Py_XDECREF(c);Py_XDECREF(b);
    }
    PyErr_Restore(type,value,trace);
}

static double point_tail(double z,double h,double m,double k,const void *table,Py_ssize_t radius) {
    double p[3];memcpy(p,(const char*)table+3*radius*sizeof(double),sizeof(p));
    double value=2/sqrt(3.141592653589793238462643383279502884)*
        (m*(p[2]/z/z+2*p[1]/z+p[0])+(k/h/z)*(p[0]+p[1]/z));
    return isfinite(value)?fmax(nextafter(0.,1.),value*(1+64*DBL_EPSILON)):NAN;
}

static PyObject *evaluate_block(PyObject *self,PyObject *args) {
    PyObject *capsule,*query,*tail_table,*rows=NULL;
    Py_buffer energies={0},tails={0};
    double h,m,k,atol,rtol,tail_share,kernel_share;
    Py_ssize_t terms,limit,maxradius;int reject_negative;
    if(!PyArg_ParseTuple(args,"OOOdddddddnnnp",&capsule,&query,&tail_table,&h,&m,&k,
        &atol,&rtol,&tail_share,&kernel_share,&terms,&limit,&maxradius,&reject_negative))return NULL;
    if(!PyCapsule_IsValid(capsule,capsule_name) || !isfinite(h) || h<=0 ||
       !isfinite(m) || m<0 || !isfinite(k) || k<0 || !isfinite(atol) || atol<=0 ||
       !isfinite(rtol) || rtol<=0 || !isfinite(tail_share) || tail_share<=0 ||
       !isfinite(kernel_share) || kernel_share<=0 || maxradius<1 || maxradius>26 ||
       terms<8 || limit<1) {
        PyErr_SetString(PyExc_ValueError,"invalid native block arguments");return NULL;
    }
    PackedPanels *packed=PyCapsule_GetPointer(capsule,capsule_name);
    if(PyObject_GetBuffer(query,&energies,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS)<0)goto done;
    if(PyObject_GetBuffer(tail_table,&tails,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS)<0)goto done;
    if(energies.ndim!=1 || energies.itemsize!=sizeof(double) || energies.format==NULL ||
       (strcmp(energies.format,"d") && strcmp(energies.format,"=d") && strcmp(energies.format,"@d")) ||
       tails.ndim!=2 || tails.shape[0]!=27 || tails.shape[1]!=3 || tails.itemsize!=sizeof(double) ||
       tails.format==NULL || strcmp(tails.format,"d")) {
        PyErr_SetString(PyExc_ValueError,"native blocks require contiguous float64 buffers");goto done;
    }
    Py_ssize_t size=energies.len/sizeof(double);
    for(int t=0;t<81;t++) {
        double cell;memcpy(&cell,(const char*)tails.buf+t*sizeof(double),sizeof(double));
        if(!isfinite(cell) || cell<0) {PyErr_SetString(PyExc_ValueError,"invalid native tail table");goto done;}
    }
    rows=PyList_New(size);if(rows==NULL)goto done;
    for(Py_ssize_t j=0;j<size;j++) {
        double energy;memcpy(&energy,(const char*)energies.buf+j*sizeof(double),sizeof(double));
        double x=sqrt(energy),z=x/h,tail;
        const char *code="unresolvable-scale";double budget=0;
        PyObject *parameters=NULL,*integral=NULL,*row=NULL;
        if(!isfinite(energy) || energy<=0 || !isfinite(z*z) || z*z==0) {
            PyErr_SetString(PyExc_ArithmeticError,"scaled energy");goto failed;
        }
        /* The first radius tried is FIRST_RADIUS = 6 Doppler widths; the tail is
         * charged to tail_share of atol + rtol*|value|, as the kernel error is,
         * and the radius grows (and the point is integrated again) only where
         * that fails. core.py:_point does the same. */
        Py_ssize_t radius=maxradius<6?maxradius:6;
        int absoluteOk;
radius_again:
        tail=point_tail(z,h,m,k,tails.buf,radius);
        if(!isfinite(tail)) {PyErr_SetString(PyExc_ArithmeticError,"tail envelope overflow");goto failed;}
        absoluteOk=tail<=tail_share*atol;
        double lower=fmax(-z,-(double)radius),upper=(double)radius;
        double low=fmax(0.,x+h*lower),high=x+h*upper;low=low*low;high=high*high;
        Py_ssize_t a=0,b=packed->size;
        while(a<b) {Py_ssize_t mid=a+(b-a)/2;if(packed->data[mid].hi<low)a=mid+1;else b=mid;}
        Py_ssize_t first=a;a=0;b=packed->size;
        while(a<b) {Py_ssize_t mid=a+(b-a)/2;if(packed->data[mid].lo<=high)a=mid+1;else b=mid;}
        Py_ssize_t last=a;
        for(int attempt=0;attempt<2;attempt++) {
            parameters=Py_BuildValue("OnnddddnnO",capsule,first,last,energy,h,lower,upper,terms,limit,
                attempt==0?Py_True:Py_False);
            if(parameters==NULL)goto failed;
            integral=integrate(self,parameters);Py_CLEAR(parameters);
            if(integral==NULL) {if(PyErr_ExceptionMatches(PyExc_OverflowError))code="resources";goto failed;}
            double value=PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(integral,0));
            double error=PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(integral,1));
            budget=kernel_share*(atol+rtol*fabs(value));
            if(error>budget) {
                Py_CLEAR(integral);
                if(attempt==0)continue;
                code="kernel-budget-exhausted";PyErr_SetString(PyExc_ArithmeticError,"analytic error estimate exceeds budget");goto failed;
            }
            if(!absoluteOk && tail>tail_share*(atol+rtol*fmax(fabs(value)-tail,0.))) {
                Py_CLEAR(integral);
                if(radius==maxradius) {
                    code="tail-budget-exhausted";PyErr_SetString(PyExc_ArithmeticError,"thermal radius limit");goto failed;
                }
                radius++;
                goto radius_again;
            }
            if(reject_negative && value<0) {
                code="negative-output";PyErr_SetString(PyExc_ArithmeticError,"recalculate with stable higher precision");goto failed;
            }
            row=Py_BuildValue("dddOOn",value,error,tail,PyTuple_GET_ITEM(integral,2),PyTuple_GET_ITEM(integral,3),radius);
            Py_CLEAR(integral);if(row==NULL)goto failed;
            PyList_SET_ITEM(rows,j,row);row=NULL;break;
        }
        if(PyErr_CheckSignals()<0) {Py_CLEAR(rows);goto done;}
        continue;
failed:
        Py_XDECREF(parameters);Py_XDECREF(integral);Py_XDECREF(row);
        if(PyErr_ExceptionMatches(PyExc_ArithmeticError) || PyErr_ExceptionMatches(PyExc_OverflowError))point_error(energy,code,budget);
        Py_CLEAR(rows);goto done;
    }
done:
    if(energies.obj!=NULL)PyBuffer_Release(&energies);
    if(tails.obj!=NULL)PyBuffer_Release(&tails);
    return rows;
}

static PyObject *special_probe(PyObject *self,PyObject *args) {
    double y;
    if(!PyArg_ParseTuple(args,"d",&y)) return NULL;
    if(!(y>=0)) {PyErr_SetString(PyExc_ValueError,"y >= 0 required");return NULL;}
    double gauss,complement=kerfc(y,&gauss);
    return Py_BuildValue("ddd",kexp(-y),complement,gauss);
}

/* Temperature-local, nonnegative source block moments. */
#define BLOCK_ORDER 16
#define BLOCK_STRIDE (BLOCK_ORDER+8)
typedef struct {
    PyObject *source;
    PackedPanels *panels;
    double *rows;
    Py_ssize_t size;
    double h;
} MomentBlocks;
static const char *block_capsule_name="kika.doppler.moment-blocks.v1";

static void free_blocks(PyObject *capsule) {
    MomentBlocks *cache=PyCapsule_GetPointer(capsule,block_capsule_name);
    if(cache){Py_XDECREF(cache->source);PyMem_Free(cache->rows);PyMem_Free(cache);}
}

static PyObject *prepare_blocks(PyObject *self,PyObject *args) {
    PyObject *source,*table,*capsule=NULL;Py_buffer b={0};double h;
    if(!PyArg_ParseTuple(args,"OOd",&source,&table,&h))return NULL;
    if(!PyCapsule_IsValid(source,capsule_name) || !isfinite(h) || h<=0) {
        PyErr_SetString(PyExc_ValueError,"invalid block source/scale");return NULL;
    }
    if(PyObject_GetBuffer(table,&b,PyBUF_FORMAT|PyBUF_ND|PyBUF_C_CONTIGUOUS)<0)return NULL;
    if(b.ndim!=2 || b.shape[1]!=BLOCK_STRIDE || b.itemsize!=sizeof(double) || !b.format ||
       (strcmp(b.format,"d") && strcmp(b.format,"=d") && strcmp(b.format,"@d")) ||
       b.shape[0]>PY_SSIZE_T_MAX/(BLOCK_STRIDE*(Py_ssize_t)sizeof(double))) {
        PyErr_SetString(PyExc_ValueError,"blocks require contiguous float64 rows");goto done;
    }
    MomentBlocks *cache=PyMem_Calloc(1,sizeof(MomentBlocks));
    if(!cache){PyErr_NoMemory();goto done;}
    cache->size=b.shape[0];cache->h=h;
    cache->panels=PyCapsule_GetPointer(source,capsule_name);
    cache->rows=PyMem_Malloc((cache->size?cache->size:1)*BLOCK_STRIDE*sizeof(double));
    if(!cache->rows){PyMem_Free(cache);PyErr_NoMemory();goto done;}
    memcpy(cache->rows,b.buf,b.len);
    Py_ssize_t previous=0;
    for(Py_ssize_t i=0;i<cache->size;i++) {
        double *r=cache->rows+i*BLOCK_STRIDE;
        for(int j=0;j<BLOCK_STRIDE;j++)if(!isfinite(r[j]))goto invalid;
        if(r[0]<0 || r[1]<=r[0] || r[2]<0 || r[3]<=0 || r[3]>.50000001 || r[6]<0 || r[7]<0 ||
           r[4]<previous || r[5]<=r[4] || r[5]>cache->panels->size || floor(r[4])!=r[4] || floor(r[5])!=r[5])goto invalid;
        Py_ssize_t first=(Py_ssize_t)r[4],last=(Py_ssize_t)r[5];
        if(r[0]!=cache->panels->data[first].lo || r[1]!=cache->panels->data[last-1].hi)goto invalid;
        for(Py_ssize_t p=first;p<last;p++) {
            NumericPanel *v=cache->panels->data+p;
            if(!v->lin || v->v0<0 || v->v1<0 || v->owner!=cache->panels->data[first].owner)goto invalid;
            if(r[7]==0 && (v->v0!=0 || v->v1!=0))goto invalid;
        }
        previous=last;
        continue;
invalid:
        PyMem_Free(cache->rows);PyMem_Free(cache);
        PyErr_SetString(PyExc_ValueError,"invalid block geometry/moments");goto done;
    }
    cache->source=Py_NewRef(source);
    capsule=PyCapsule_New(cache,block_capsule_name,free_blocks);
    if(!capsule){Py_DECREF(cache->source);PyMem_Free(cache->rows);PyMem_Free(cache);goto done;}
    capsule=Py_BuildValue("Nn",capsule,sizeof(MomentBlocks)+cache->size*BLOCK_STRIDE*(Py_ssize_t)sizeof(double));
done:
    PyBuffer_Release(&b);return capsule;
}

static PyObject *block_values_cached(PyObject *self,PyObject *args) {
    PyObject *capsule,*query,*result=NULL;Py_buffer q={0};
    if(!PyArg_ParseTuple(args,"OO",&capsule,&query))return NULL;
    if(!PyCapsule_IsValid(capsule,block_capsule_name)) {
        PyErr_SetString(PyExc_ValueError,"prepared moment blocks required");return NULL;
    }
    MomentBlocks *cache=PyCapsule_GetPointer(capsule,block_capsule_name);
    if(PyObject_GetBuffer(query,&q,PyBUF_FORMAT|PyBUF_ND|PyBUF_C_CONTIGUOUS)<0)return NULL;
    if(q.ndim!=1 || q.itemsize!=sizeof(double) || !q.format ||
       (strcmp(q.format,"d") && strcmp(q.format,"=d") && strcmp(q.format,"@d"))) {
        PyErr_SetString(PyExc_ValueError,"query requires contiguous float64 buffer");goto done;
    }
    result=PyList_New(q.shape[0]);if(!result)goto done;
    double h=cache->h;
    for(Py_ssize_t i=0;i<q.shape[0];i++) {
        double e;memcpy(&e,(const char*)q.buf+i*sizeof(double),sizeof(double));
        if(!isfinite(e) || e<=0) {PyErr_SetString(PyExc_ValueError,"positive finite query required");Py_CLEAR(result);goto done;}
        double x=sqrt(e),z=x/h,low=fmax(0.,x+h*fmax(-z,-6.));low*=low;
        double high=x+6*h;high*=high;
        Sum value={0},error={0};PyObject *spans=PyList_New(0),*components=PyDict_New();
        if(!spans || !components){Py_XDECREF(spans);Py_XDECREF(components);Py_CLEAR(result);goto done;}
        Py_ssize_t a=0,c=cache->size,run_first=-1,run_last=-1,used=0;
        PyObject *owner=NULL;double component_value=0.;
        while(a<c){Py_ssize_t mid=(a+c)/2;if(cache->rows[mid*BLOCK_STRIDE+1]<low)a=mid+1;else c=mid;}
        for(Py_ssize_t j=a;j<cache->size && cache->rows[j*BLOCK_STRIDE]<=high;j++) {
            const double *r=cache->rows+j*BLOCK_STRIDE;
            double s[2]={(r[2]-x)/h,(r[2]+x)/h};Sum integral={0},bounds={0};
            for(int side=0;side<2;side++) {
                if(side && s[side]-r[3]>8.) {
                    add(&bounds,kgauss(s[side]-r[3])*(r[7]+r[6])*(1+64*DBL_EPSILON));
                    continue;
                }
                double g=kgauss(fabs(s[side]));if(g==0)continue;
                double A=-2*s[side]*r[3],B=-2*r[3]*r[3],prior=0.,cn=1.,norm=1.;Sum contracted={0};
                add(&contracted,r[7]);
                for(int k=1;k<=BLOCK_ORDER;k++) {
                    double next=(A*cn+B*prior)/k;prior=cn;cn=next;
                    norm+=fabs(cn);add(&contracted,cn*r[7+k]);
                }
                double rem=coefficient_tail(fabs(A),fabs(B),fabs(cn),fabs(prior),BLOCK_ORDER);
                add(&integral,(side?-1.:1.)*g*total(contracted));
                add(&bounds,g*((r[7]+r[6])*rem+r[6]*norm+512*DBL_EPSILON*r[7]*norm));
            }
            double scale=1./(sqrt(3.14159265358979323846)*h*e);
            double v=scale*total(integral);add(&value,v);add(&error,scale*total(bounds));used++;
            Py_ssize_t first=(Py_ssize_t)r[4],last=(Py_ssize_t)r[5];
            PyObject *current=cache->panels->data[first].owner;
            if(owner && owner!=current){if(store_component(components,owner,component_value)<0)goto failed;component_value=0.;}
            owner=current;component_value+=v;
            if(run_last==first)run_last=last;
            else {
                if(run_first>=0){PyObject *span=Py_BuildValue("nn",run_first,run_last);if(!span)goto failed;
                    int rc=PyList_Append(spans,span);Py_DECREF(span);if(rc<0)goto failed;}
                run_first=first;run_last=last;
            }
        }
        if(owner && store_component(components,owner,component_value)<0)goto failed;
        if(run_first>=0){PyObject *span=Py_BuildValue("nn",run_first,run_last);if(!span)goto failed;
            int rc=PyList_Append(spans,span);Py_DECREF(span);if(rc<0)goto failed;}
        PyObject *row=Py_BuildValue("ddNNn",total(value),total(error),spans,components,used);
        if(!row){Py_CLEAR(result);goto done;}
        PyList_SET_ITEM(result,i,row);
        if(PyErr_CheckSignals()<0){Py_CLEAR(result);goto done;}
        continue;
failed:
        Py_DECREF(spans);Py_DECREF(components);Py_CLEAR(result);goto done;
    }
done:
    PyBuffer_Release(&q);return result;
}

static PyMethodDef methods[]={
    {"prepare_blocks",prepare_blocks,METH_VARARGS,"Prepare temperature-local polynomial moments."},
    {"block_values_cached",block_values_cached,METH_VARARGS,"Evaluate immutable source block moments."},
    {"prepare_panels",prepare_panels,METH_O,"Prepare immutable numeric source panels."},
    {"evaluate_block",evaluate_block,METH_VARARGS,"Evaluate bounded point blocks with original budgets."},
    {"integrate",integrate,METH_VARARGS,"Integrate validated analytic source panels."},
    {"moments",moment_probe,METH_VARARGS,"Private recurrence and remainder diagnostic."},
    {"closed_moments",closed_probe,METH_VARARGS,"Private closed-moment conditioning diagnostic."},
    {"special",special_probe,METH_VARARGS,"Private kexp(-y), erfc(y), exp(-y*y) diagnostic."},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_doppler_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__doppler_native(void) {
    for(int n=0;n<80;n++) for(int j=0;j<5;j++) weights[n][j]=1./(n+j+1);
    return PyModule_Create(&module);
}
