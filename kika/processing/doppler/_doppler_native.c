/* Analytic Doppler panel integration, strict float64, no NumPy C ABI.
 * Same local Taylor recurrence and budgets as the Python reference.
 * No fast math, thinning, changed support or sampled integration.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

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
    double plus=exp(-s*s),t=z+s,minus=plus*exp(-4*z*t);
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
            if(terms==80 && n>=8) {
                double bound=coefficient_tail(2*(fabs(t)+z)*d,2*d*d,
                    fabs(dif[n])+fabs(other[n]),fabs(dif[n-1])+fabs(other[n-1]),n);
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
            if(terms==80 && n>=8) {
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

static PyObject *integrate(PyObject *self,PyObject *args) {
    PyObject *panels,*components=NULL,*result=NULL;
    Py_ssize_t first,last,terms,limit,count=0;
    double energy,h,lower,upper;
    double *work=NULL;
    if(!PyArg_ParseTuple(args,"Onnddddnn",&panels,&first,&last,&energy,&h,&lower,&upper,&terms,&limit)) return NULL;
    if(!PyTuple_Check(panels) || first<0 || last<first || last>PyTuple_GET_SIZE(panels) ||
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
    Sum values={0},errors={0},absolute={0};
    for(Py_ssize_t p=first;p<last;p++) {
        PyObject *panel=PyTuple_GET_ITEM(panels,p);
        if(!PyTuple_Check(panel) || PyTuple_GET_SIZE(panel)!=7) {
            PyErr_SetString(PyExc_ValueError,"invalid native Doppler panel");goto done;
        }
        double lo=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,0));
        double hi=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,1));
        double v0=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,2));
        double v1=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,3));
        double anchor=PyFloat_AsDouble(PyTuple_GET_ITEM(panel,6));
        PyObject *kind=PyTuple_GET_ITEM(panel,4),*owner=PyTuple_GET_ITEM(panel,5);
        if(PyErr_Occurred()) goto done;
        int lin=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"linlin")==0;
        int inv=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"one_over_v")==0;
        int constant=PyUnicode_Check(kind) && PyUnicode_CompareWithASCIIString(kind,"constant")==0;
        if(!isfinite(lo) || lo<0 || !(hi>lo) || !isfinite(v0) || !isfinite(v1) ||
           (!lin && !inv && !constant) || (lin && !isfinite(hi)) ||
           (inv && (!isfinite(anchor) || anchor<=0)) || !PyUnicode_Check(owner)) {
            PyErr_SetString(PyExc_ValueError,"invalid native Doppler panel"); goto done;
        }
        int include=0;
        Sum contribution={0};
        if(v0==0 && v1==0) include=1;
        else {
            double ylo=sqrt(lo),yhi=sqrt(hi);
            double slo=(lo-energy)/(h*(ylo+x));
            double shi=isinf(hi)?INFINITY:(hi-energy)/(h*(yhi+x));
            double left=fmax(lower,slo),right=fmin(upper,shi);
            if(left>=right) continue;
            double ds=isinf(hi)?INFINITY:(hi-lo)/(h*(yhi+ylo));
            int narrow=ds<1;
            if(ds<=0) {PyErr_SetString(PyExc_ArithmeticError,"panel overflow");goto done;}
            double qleft=narrow?fmax(0,(lower-slo)/ds):0;
            double qright=narrow?fmin(1,(upper-slo)/ds):1;
            double width=narrow?(qright-qleft)*ds:right-left;
            double subdivisions=fmax(1,ceil(width/.25));
            if(!isfinite(subdivisions) || subdivisions>limit-count) {
                PyErr_SetString(PyExc_OverflowError,"analytic subpanel limit");goto done;
            }
            Py_ssize_t nsub=(Py_ssize_t)subdivisions;
            for(Py_ssize_t i=0;i<nsub;i++) {
                double d=width/nsub;
                double s=narrow?slo+(qleft+i*(qright-qleft)/nsub)*ds:left+i*d;
                double yr=1+s/z,dr=d/z,poly[5]={0},sig[3]={0}; int ncoeff;
                if(inv) {
                    poly[0]=v0*sqrt(anchor)/x*yr;poly[1]=v0*sqrt(anchor)/x*dr;ncoeff=2;
                } else {
                    int nsig;
                    if(lin) {
                        double sig0,slope=(v1-v0)/(hi-lo);
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
                moments(s,d,z,terms,work,work+terms,mom,&rem,&norm,ncoeff,poly);
                Sum pnorm={0};
                for(int j=0;j<ncoeff;j++) add(&pnorm,fabs(poly[j]));
                double factor=d/sqrt(3.141592653589793238462643383279502884);
                double value=factor*mom[0],err=factor*total(pnorm)*(rem+256*DBL_EPSILON*norm);
                if(!isfinite(value) || !isfinite(err)) {PyErr_SetString(PyExc_ArithmeticError,"panel overflow");goto done;}
                add(&values,value);add(&errors,err);add(&absolute,fabs(value));add(&contribution,value);
                count++;include=1;
            }
        }
        if(include) {
            PyObject *previous=PyDict_GetItemWithError(components,owner);
            if(previous==NULL && PyErr_Occurred()) goto done;
            double prior=previous?PyFloat_AsDouble(previous):0;
            PyObject *v=PyFloat_FromDouble(prior+total(contribution));
            if(v==NULL) goto done;
            int rc=PyDict_SetItem(components,owner,v);Py_DECREF(v);if(rc<0) goto done;
        }
        if((p-first)%1024==0 && PyErr_CheckSignals()<0) goto done;
    }
    double value=total(values),error=total(errors)+8*DBL_EPSILON*total(absolute);
    if(!isfinite(value) || !isfinite(error)) {
        PyErr_SetString(PyExc_ArithmeticError,"aggregate overflow");goto done;
    }
    result=Py_BuildValue("ddOn",value,error,components,count);
done:
    Py_XDECREF(components);PyMem_Free(work);return result;
}
static PyMethodDef methods[]={
    {"integrate",integrate,METH_VARARGS,"Integrate validated analytic source panels."},
    {"moments",moment_probe,METH_VARARGS,"Private recurrence and remainder diagnostic."},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_doppler_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__doppler_native(void) {
    for(int n=0;n<80;n++) for(int j=0;j<5;j++) weights[n][j]=1./(n+j+1);
    return PyModule_Create(&module);
}
