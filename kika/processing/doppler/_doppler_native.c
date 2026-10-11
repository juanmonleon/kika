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
            if(terms==80 && n>=(poly==NULL?8:2)) {
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
    double lo,hi,v0,v1,anchor,ylo,yhi,slope;
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
    Sum values={0},errors={0},absolute={0};
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
            if(!isfinite(subdivisions) || subdivisions>limit-reserved) {
                PyErr_SetString(PyExc_OverflowError,"analytic subpanel limit");goto done;
            }
            Py_ssize_t planned=(Py_ssize_t)subdivisions;reserved+=planned;
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
                add(&values,value);add(&errors,err);add(&absolute,fabs(value));add(&contribution,value);
                count++;include=1;
            }
        }
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
    double value=total(values),error=total(errors)+8*DBL_EPSILON*total(absolute);
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

static PyMethodDef methods[]={
    {"prepare_panels",prepare_panels,METH_O,"Prepare immutable numeric source panels."},
    {"evaluate_block",evaluate_block,METH_VARARGS,"Evaluate bounded point blocks with original budgets."},
    {"integrate",integrate,METH_VARARGS,"Integrate validated analytic source panels."},
    {"moments",moment_probe,METH_VARARGS,"Private recurrence and remainder diagnostic."},
    {"closed_moments",closed_probe,METH_VARARGS,"Private closed-moment conditioning diagnostic."},
    {NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_doppler_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__doppler_native(void) {
    for(int n=0;n<80;n++) for(int j=0;j<5;j++) weights[n][j]=1./(n+j+1);
    return PyModule_Create(&module);
}
