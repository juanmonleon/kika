/* Charged-channel functions of kika.processing.resonances.coulomb, per row.
 *
 * open_rows: the rho>0.25, eta>0 branch of coulomb._charged_point. Outside
 * the barrier (P>=|S|/5), Steed's outgoing continued fraction at rho. Inside it, S from
 * that fraction at rho and ln(P) from the Wronskian with F's power series;
 * where that is not well conditioned, the fraction at turning+8 and the
 * scaled Riccati transport to rho, with the DOP853 tableau, error norm, step
 * control and initial step of scipy.integrate.solve_ivp (the coefficients
 * are passed in). The regular continued fraction at rho gives the phase.
 * Below rho=0.25 with eta*rho>2, the fraction and the Wronskian while the
 * fraction converges within SMALL_RADIUS_FRACTION_ITERATIONS.
 *
 * closed_rows: coulomb._closed_fraction where it converges within its
 * iteration limit; elsewhere the Laplace integral of
 * coulomb.closed_charged_shift, by the trapezoidal rule in the logarithm of the integration variable, halving the
 * step until two levels agree. The integrand is analytic in a strip around
 * the real axis, so the rule converges geometrically.
 *
 * A row that fails any check gets ok=0 and is left to the Python reference,
 * which either computes it or raises its own error. Float64 only, no
 * fast-math.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifdef __FAST_MATH__
#error "Coulomb functions require strict floating-point arithmetic"
#endif
typedef char cf_requires_ieee_float64[
    (sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024)?1:-1];

#define TINY 1e-150
#define FRACTION_ITERATIONS 20000
#define TRANSPORT_EVALUATIONS 100000
#define STAGES 12
#define CLOSED_LEVELS 22
#define CLOSED_SIDE_STEPS 100000
#define CLOSED_FRACTION_ITERATIONS 500
#define SMALL_RADIUS_FRACTION_ITERATIONS 3000

typedef struct {double re,im;} cx;

static cx cx_make(double re,double im){cx z;z.re=re;z.im=im;return z;}
static cx cx_add(cx a,cx b){return cx_make(a.re+b.re,a.im+b.im);}
static cx cx_mul(cx a,cx b){return cx_make(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);}
static double cx_abs(cx a){return hypot(a.re,a.im);}

/* Smith's division, as CPython's complex quotient. */
static cx cx_div(cx a,cx b){
    double abs_r=fabs(b.re),abs_i=fabs(b.im);
    if(abs_r>=abs_i){
        if(abs_r==0.)return cx_make(a.re/abs_r,a.im/abs_r);
        double ratio=b.im/b.re,denom=b.re+b.im*ratio;
        return cx_make((a.re+a.im*ratio)/denom,(a.im-a.re*ratio)/denom);
    }
    if(abs_i>=abs_r){
        double ratio=b.re/b.im,denom=b.re*ratio+b.im;
        return cx_make((a.re*ratio+a.im)/denom,(a.im*ratio-a.re)/denom);
    }
    return cx_make(NAN,NAN);
}

/* coulomb._outgoing: rho*H'/H of the outgoing wave, modified Lentz. */
static int outgoing(int l,double eta,double rho,int limit,cx *out){
    cx a=cx_make(l+1,eta),b=cx_make(-l,eta);
    cx f=cx_make(0.,1.-eta/rho);
    if(f.re==0.&&f.im==0.)f=cx_make(TINY,0.);
    cx c=f,d=cx_make(0.,0.);int stable=0;
    for(int n=1;n<=limit;n++){
        cx an=n==1?cx_mul(cx_mul(cx_make(0.,1./rho),a),b)
                  :cx_mul(cx_make(a.re+n-1,a.im),cx_make(b.re+n-1,b.im));
        cx bn=cx_make(2*(rho-eta),2.*n);
        d=cx_add(bn,cx_mul(an,d));c=cx_add(bn,cx_div(an,c));
        if(cx_abs(d)<TINY)d=cx_make(TINY,0.);
        if(cx_abs(c)<TINY)c=cx_make(TINY,0.);
        d=cx_div(cx_make(1.,0.),d);
        cx change=cx_mul(c,d);f=cx_mul(f,change);
        stable=cx_abs(cx_make(change.re-1.,change.im))<2e-14?stable+1:0;
        if(stable>=3){
            if(!isfinite(f.re)||!isfinite(f.im))return 0;
            *out=f;return 1;
        }
    }
    return 0;
}

/* coulomb._regular: rho*F'/F, the same fraction in real arithmetic. */
static int regular(int l,double eta,double rho,double *out){
    int first=l+1;
    double f=first/rho+eta/first;
    if(f==0.)f=TINY;
    double c=f,d=0.;int stable=0;
    for(int n=1;n<=FRACTION_ITERATIONS;n++){
        int k=l+n;double q=eta/k;
        double an=-(1+q*q),bn=(2*k+1)/rho+eta/k+eta/(k+1);
        d=bn+an*d;c=bn+an/c;
        if(fabs(d)<TINY)d=TINY;
        if(fabs(c)<TINY)c=TINY;
        d=1/d;double change=c*d;f*=change;
        stable=fabs(change-1)<2e-14?stable+1:0;
        if(stable>=3){
            if(!isfinite(f))return 0;
            *out=f;return 1;
        }
    }
    return 0;
}

typedef struct {
    const double *a,*b,*c,*e3,*e5;   /* a: STAGES x STAGES, e3/e5: STAGES+1 */
} tableau;

static void riccati(double t,const double *y,double eta,double centrifugal,double *f){
    double rho=exp(t),s=y[0],p=exp(y[1]);
    f[0]=s-rho*rho+2*eta*rho+centrifugal-s*s+p*p;
    f[1]=1-2*s;
}

static double rms2(double a,double b){return sqrt(a*a+b*b)/sqrt(2.);}

/* coulomb._transport: solve_ivp(..., method='DOP853', rtol=2e-12, atol=2e-13). */
static int transport(int l,double eta,double start,double stop,double *y,const tableau *tb){
    const double rtol=2e-12,atol=2e-13;
    double centrifugal=(double)l*(l+1);
    double t=log(start),bound=log(stop);
    double direction=bound>t?1.:(bound<t?-1.:0.);
    double f[2],f1[2],y1[2],K[STAGES+1][2],stage[2],y_new[2],f_new[2];
    long evaluations=0;
    riccati(t,y,eta,centrifugal,f);evaluations++;
    /* select_initial_step with order 7 */
    double interval=fabs(bound-t);
    if(interval==0.)return 1;
    double sc0=atol+fabs(y[0])*rtol,sc1=atol+fabs(y[1])*rtol;
    double d0=rms2(y[0]/sc0,y[1]/sc1),d1=rms2(f[0]/sc0,f[1]/sc1),h0;
    h0=(d0<1e-5||d1<1e-5)?1e-6:.01*d0/d1;
    if(h0>interval)h0=interval;
    y1[0]=y[0]+h0*direction*f[0];y1[1]=y[1]+h0*direction*f[1];
    riccati(t+h0*direction,y1,eta,centrifugal,f1);evaluations++;
    double d2=rms2((f1[0]-f[0])/sc0,(f1[1]-f[1])/sc1)/h0,h1;
    if(d1<=1e-15&&d2<=1e-15)h1=fmax(1e-6,h0*1e-3);
    else h1=pow(.01/fmax(d1,d2),1./8);
    double h_abs=fmin(fmin(100*h0,h1),interval);
    while(direction*(t-bound)<0){
        double min_step=10*fabs(nextafter(t,direction*INFINITY)-t);
        if(h_abs<min_step)h_abs=min_step;
        int rejected=0,accepted=0;
        double t_new=t;
        while(!accepted){
            if(h_abs<min_step)return 0;
            double h=h_abs*direction;t_new=t+h;
            if(direction*(t_new-bound)>0)t_new=bound;
            h=t_new-t;h_abs=fabs(h);
            K[0][0]=f[0];K[0][1]=f[1];
            for(int s=1;s<STAGES;s++){
                const double *as=tb->a+s*STAGES;
                double dy0=0.,dy1=0.;
                for(int j=0;j<s;j++){dy0+=K[j][0]*as[j];dy1+=K[j][1]*as[j];}
                stage[0]=y[0]+dy0*h;stage[1]=y[1]+dy1*h;
                riccati(t+tb->c[s]*h,stage,eta,centrifugal,K[s]);
            }
            double s0=0.,s1=0.;
            for(int j=0;j<STAGES;j++){s0+=K[j][0]*tb->b[j];s1+=K[j][1]*tb->b[j];}
            y_new[0]=y[0]+h*s0;y_new[1]=y[1]+h*s1;
            riccati(t+h,y_new,eta,centrifugal,f_new);
            K[STAGES][0]=f_new[0];K[STAGES][1]=f_new[1];
            evaluations+=STAGES;
            if(evaluations>TRANSPORT_EVALUATIONS)return 0;
            double scale0=atol+fmax(fabs(y[0]),fabs(y_new[0]))*rtol;
            double scale1=atol+fmax(fabs(y[1]),fabs(y_new[1]))*rtol;
            double e50=0.,e51=0.,e30=0.,e31=0.;
            for(int j=0;j<=STAGES;j++){
                e50+=K[j][0]*tb->e5[j];e51+=K[j][1]*tb->e5[j];
                e30+=K[j][0]*tb->e3[j];e31+=K[j][1]*tb->e3[j];
            }
            e50/=scale0;e51/=scale1;e30/=scale0;e31/=scale1;
            double n5=e50*e50+e51*e51,n3=e30*e30+e31*e31,error;
            if(n5==0.&&n3==0.)error=0.;
            else error=h_abs*n5/sqrt((n5+.01*n3)*2);
            if(!isfinite(error)||!isfinite(y_new[0])||!isfinite(y_new[1]))return 0;
            if(error<1){
                double factor=error==0.?10.:fmin(10.,.9*pow(error,-1./8));
                if(rejected)factor=fmin(1.,factor);
                h_abs*=factor;accepted=1;
            }else{
                h_abs*=fmax(.2,.9*pow(error,-1./8));rejected=1;
            }
        }
        t=t_new;y[0]=y_new[0];y[1]=y_new[1];f[0]=f_new[0];f[1]=f_new[1];
    }
    return 1;
}

/* ln|Gamma(l+1+i eta)|, eta>0: |Gamma(1+i eta)|^2 = pi eta/sinh(pi eta). */
static double log_abs_gamma(int l,double eta){
    double x=M_PI*eta;
    double log_sinh=x>1?x+log1p(-exp(-2*x))-log(2.):log(sinh(x));
    double value=.5*(log(x)-log_sinh);
    for(int k=1;k<=l;k++)value+=.5*log((double)k*k+eta*eta);
    return value;
}

/* coulomb._barrier_log_p: ln(P) from F's series, F'/F and Re(H'/H). */
static int barrier_log_p(int l,double eta,double rho,cx here,double reg,double *out){
    double log_c=l*log(2.)-.5*M_PI*eta+log_abs_gamma(l,eta)-lgamma(2.*l+2);
    double previous=0.,term=1.,total=1.,magnitude=1.;int quiet=0,j;
    for(j=1;j<10000;j++){
        double next=(2*eta*rho*term-rho*rho*previous)/((double)j*(2*l+1+j));
        previous=term;term=next;
        total+=term;magnitude+=fabs(term);
        if(!isfinite(total)||total>1e280)return 0;
        quiet=fabs(term)<1e-17*fabs(total)?quiet+1:0;
        if(quiet>=2)break;
    }
    if(j>=10000||total<=0||magnitude>8*total)return 0;
    double log_f=log_c+(l+1)*log(rho)+log(total);
    double a=here.re-reg;
    if(!(a<0))return 0;
    double disc=1-exp(log(4*a*a)+4*log_f);
    if(!(disc>=.25))return 0;
    double log_g=-log_f+log1p(sqrt(disc))-log(2*fabs(a));
    *out=log(rho)-2*log_g-log1p(exp(2*log_f-2*log_g));
    return isfinite(*out);
}

/* coulomb._charged_point, branch rho<=0.25 and eta*rho>2: the fraction and
 * the Wronskian, while the fraction stays within its iteration limit. */
static int small_point(int l,double eta,double rho,double *log_p,double *shift,double *phase){
    cx here;double reg;
    if(!outgoing(l,eta,rho,SMALL_RADIUS_FRACTION_ITERATIONS,&here))return 0;
    if(!regular(l,eta,rho,&reg)||!barrier_log_p(l,eta,rho,here,reg,log_p))return 0;
    *shift=rho*here.re;
    if(!isfinite(*shift))return 0;
    *phase=atan2(exp(*log_p),rho*reg-*shift);
    return isfinite(*phase);
}

/* coulomb._charged_point, branch rho>0.25 and eta>0. */
static int open_point(int l,double eta,double rho,const tableau *tb,
                      double *log_p,double *shift,double *phase){
    double turning=eta+hypot(eta,sqrt((double)l*(l+1)));
    double start=fmax(rho,turning+8.);
    double y[2],reg;int have_reg=0,done=0;
    if(start!=rho){
        cx here;
        if(outgoing(l,eta,rho,FRACTION_ITERATIONS,&here)){
            if(here.im>0&&rho*here.im>=.2*fabs(rho*here.re)){
                y[0]=rho*here.re;y[1]=log(rho)+log(here.im);done=1;
            }else if(regular(l,eta,rho,&reg)){
                have_reg=1;
                if(barrier_log_p(l,eta,rho,here,reg,&y[1])){y[0]=rho*here.re;done=1;}
            }
        }
    }
    if(!done){
        cx derivative;
        if(!outgoing(l,eta,start,FRACTION_ITERATIONS,&derivative)||derivative.im<=0)return 0;
        y[0]=start*derivative.re;y[1]=log(start)+log(derivative.im);
        if(start!=rho&&!transport(l,eta,start,rho,y,tb))return 0;
    }
    if(!isfinite(y[0]+y[1]))return 0;
    if(!have_reg&&!regular(l,eta,rho,&reg))return 0;
    *shift=y[0];*log_p=y[1];
    *phase=atan2(exp(y[1]),rho*reg-y[0]);
    return isfinite(*phase);
}

/* Neumaier-compensated accumulator. */
typedef struct {double sum,carry;} acc;
static void acc_add(acc *a,double v){
    double t=a->sum+v;
    if(fabs(a->sum)>=fabs(v))a->carry+=(a->sum-t)+v;
    else a->carry+=(v-t)+a->sum;
    a->sum=t;
}
static double acc_value(const acc *a){return a->sum+a->carry;}

typedef struct {int l;double eta,kappa,mode,peak;} closed_problem;

static double closed_log_weight(const closed_problem *p,double u){
    return 2*p->l*log(u)-(p->eta-p->l)*log1p(p->kappa/u)-2*u;
}

/* Integrands of norm and moment in x=ln(v): v^(k+1) w(v). */
static void closed_terms(const closed_problem *p,double x,double *g0,double *g1){
    double v=exp(x);
    double w=exp(closed_log_weight(p,p->mode*v)-p->peak);
    *g0=w*v;*g1=w*v*v;
}

/* coulomb._closed_fraction: Steed's fraction at rho=i*kappa, in real form. */
static int closed_fraction(int l,double eta,double kappa,double *out){
    double a=l+1+eta,b=-l+eta;
    double phi=1+eta/kappa,gamma=phi,delta=0.;int stable=0;
    for(int n=1;n<=CLOSED_FRACTION_ITERATIONS;n++){
        double an=n==1?a*b/kappa:(a+n-1)*(b+n-1);
        double beta=2*(kappa+eta+n);
        delta=beta-an*delta;gamma=beta-an/gamma;
        if(fabs(delta)<TINY)delta=TINY;
        if(fabs(gamma)<TINY)gamma=TINY;
        delta=1/delta;double change=gamma*delta;phi*=change;
        stable=fabs(change-1)<2e-14?stable+1:0;
        if(stable>=3){*out=-kappa*phi;return isfinite(*out);}
    }
    return 0;
}

/* coulomb.closed_charged_shift at one point, eta>0. */
static int closed_point(int l,double eta,double kappa,double *out){
    if(closed_fraction(l,eta,kappa,out))return 1;
    closed_problem p;
    p.l=l;p.eta=eta;p.kappa=kappa;
    double lr=l-kappa;
    p.mode=.5*(lr+sqrt(lr*lr+2*(eta+l)*kappa));
    if(p.mode<=0)p.mode=1.;
    p.peak=closed_log_weight(&p,p.mode);
    if(!isfinite(p.peak))return 0;
    /* Step from the curvature of ln w at the mode, in x=ln(v). */
    double u=p.mode,r=kappa,hl=eta-l;
    double d1=2*l/u+hl*r/(u*(u+r))-2;
    double d2=-2*l/(u*u)-hl*r*(2*u+r)/(u*u*(u+r)*(u+r));
    double curvature=-(u*d1+u*u*d2);
    double step=.5;
    if(isfinite(curvature)&&curvature>0)step=fmin(.5,.5/sqrt(curvature));
    /* Truncate each side where both integrands fall below 1e-20 of the
     * largest value seen: the tails decay at least exponentially in x, so
     * the omitted mass is far below the 1e-14 the levels must agree to. */
    double g0,g1,top0=0.,top1=0.;
    closed_terms(&p,0.,&g0,&g1);top0=g0;top1=g1;
    double lo=0.,hi=0.;
    for(int side=-1;side<=1;side+=2){
        int quiet=0;
        for(long k=1;;k++){
            if(k>CLOSED_SIDE_STEPS)return 0;
            double x=side*k*step;
            closed_terms(&p,x,&g0,&g1);
            if(!isfinite(g0)||!isfinite(g1))return 0;
            if(g0>top0)top0=g0;
            if(g1>top1)top1=g1;
            if(g0<1e-20*top0&&g1<1e-20*top1){if(++quiet>=2){if(side<0)lo=x;else hi=x;break;}}
            else quiet=0;
        }
    }
    long n=(long)((hi-lo)/step+.5);
    acc s0={0.,0.},s1={0.,0.};
    for(long k=0;k<=n;k++){
        closed_terms(&p,lo+k*step,&g0,&g1);
        double wgt=(k==0||k==n)?.5:1.;
        acc_add(&s0,wgt*g0);acc_add(&s1,wgt*g1);
    }
    double t0=acc_value(&s0)*step,t1=acc_value(&s1)*step;
    for(int level=0;level<CLOSED_LEVELS;level++){
        double h=step/2;
        for(long k=0;k<n;k++){
            closed_terms(&p,lo+(2*k+1)*h,&g0,&g1);
            acc_add(&s0,g0);acc_add(&s1,g1);
        }
        n*=2;step=h;
        double n0=acc_value(&s0)*step,n1=acc_value(&s1)*step;
        if(!(n0>0)||!isfinite(n0+n1))return 0;
        int converged=fabs(n0-t0)<=1e-14*n0&&fabs(n1-t1)<=1e-14*n1;
        t0=n0;t1=n1;
        /* Geometric convergence: once two levels agree, the finer one is
         * accurate well beyond their difference. */
        if(converged){
            *out=l+1-kappa-2*p.mode*t1/t0;
            return isfinite(*out);
        }
    }
    return 0;
}

static int get(PyObject *obj,Py_buffer *b,Py_ssize_t count,const char *format,int writable){
    int flags=PyBUF_FORMAT|PyBUF_C_CONTIGUOUS;
    if(writable)flags|=PyBUF_WRITABLE;
    if(PyObject_GetBuffer(obj,b,flags)<0)return 0;
    Py_ssize_t item=strcmp(format,"d")==0?8:1;
    if(((uintptr_t)b->buf%(uintptr_t)item)!=0||b->ndim!=1||b->itemsize!=item||b->format==NULL||
       strcmp(b->format,format)!=0||(count>=0&&b->len!=count*item)){
        PyErr_SetString(PyExc_ValueError,"native Coulomb requires exact contiguous float64/uint8 buffers");
        return 0;
    }
    return 1;
}

static void release_all(Py_buffer *b,int n){
    for(int i=0;i<n;i++)if(b[i].obj!=NULL)PyBuffer_Release(&b[i]);
}

static PyObject *open_rows(PyObject *self,PyObject *args){
    int l;Py_ssize_t n;
    PyObject *o[11];Py_buffer b[11];memset(b,0,sizeof b);
    if(!PyArg_ParseTuple(args,"inOOOOOOOOOOO",&l,&n,&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&o[6],
                         &o[7],&o[8],&o[9],&o[10]))return NULL;
    if(l<0||l>64||n<0){PyErr_SetString(PyExc_ValueError,"invalid native Coulomb dimensions");return NULL;}
    Py_ssize_t counts[11]={n,n,STAGES*STAGES,STAGES,STAGES,STAGES+1,STAGES+1,n,n,n,n};
    const char *formats[11]={"d","d","d","d","d","d","d","d","d","d","B"};
    int writable[11]={0,0,0,0,0,0,0,1,1,1,1};
    for(int i=0;i<11;i++)if(!get(o[i],&b[i],counts[i],formats[i],writable[i])){release_all(b,11);return NULL;}
    const double *eta=b[0].buf,*rho=b[1].buf;
    tableau tb={b[2].buf,b[3].buf,b[4].buf,b[5].buf,b[6].buf};
    double *log_p=b[7].buf,*shift=b[8].buf,*phase=b[9].buf;unsigned char *ok=b[10].buf;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t i=0;i<n;i++)
        ok[i]=(unsigned char)(eta[i]>0&&(rho[i]>.25?open_point(l,eta[i],rho[i],&tb,&log_p[i],&shift[i],&phase[i])
                                       :eta[i]*rho[i]>2.&&small_point(l,eta[i],rho[i],&log_p[i],&shift[i],&phase[i])));
    Py_END_ALLOW_THREADS
    release_all(b,11);
    Py_RETURN_NONE;
}

static PyObject *closed_rows(PyObject *self,PyObject *args){
    int l;Py_ssize_t n;
    PyObject *o[4];Py_buffer b[4];memset(b,0,sizeof b);
    if(!PyArg_ParseTuple(args,"inOOOO",&l,&n,&o[0],&o[1],&o[2],&o[3]))return NULL;
    if(l<0||l>64||n<0){PyErr_SetString(PyExc_ValueError,"invalid native Coulomb dimensions");return NULL;}
    const char *formats[4]={"d","d","d","B"};int writable[4]={0,0,1,1};
    for(int i=0;i<4;i++)if(!get(o[i],&b[i],n,formats[i],writable[i])){release_all(b,4);return NULL;}
    const double *eta=b[0].buf,*kappa=b[1].buf;double *shift=b[2].buf;unsigned char *ok=b[3].buf;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t i=0;i<n;i++)
        ok[i]=(unsigned char)(eta[i]>0&&kappa[i]>0&&closed_point(l,eta[i],kappa[i],&shift[i]));
    Py_END_ALLOW_THREADS
    release_all(b,4);
    Py_RETURN_NONE;
}

static PyMethodDef methods[]={
    {"open_rows",open_rows,METH_VARARGS,"ln(P), S and phase of open charged channels outside the small-radius series; ok=0 rows need the reference."},
    {"closed_rows",closed_rows,METH_VARARGS,"Shift of closed charged channels; ok=0 rows need the reference."},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_coulomb_native",NULL,0,methods};
PyMODINIT_FUNC PyInit__coulomb_native(void){
    PyObject *m=PyModule_Create(&module);
    if(m!=NULL&&PyModule_AddIntConstant(m,"API_VERSION",1)<0){Py_DECREF(m);return NULL;}
    return m;
}
