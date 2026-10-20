/* The URR ladder kernel (urr_ladder), included once per instruction set.
 *
 * URR_NAME(x) names this copy's functions; URR_TARGET is its target
 * attribute (empty for the baseline). The arithmetic is humlicek's W4, region
 * by region, with three changes that move no result beyond rounding (2e-14 of
 * the largest value on a U-235-like ladder):
 *
 * - x' = (E - Er)/Delta is formed with one multiplication by 1/Delta instead of
 *   2(E - Er)/Gamma times Gamma/(2 Delta);
 * - a complex division is one reciprocal of |d|^2 (|t| < 30 here, so |d|^2
 *   neither overflows nor underflows) instead of Smith's three divisions;
 * - the samples of a resonance, sorted, cross W4's regions in runs (the
 *   region depends on |x'| alone at fixed y), and each run is one loop over a
 *   single branch-free formula, which the compiler vectorizes. The region of
 *   every sample is decided by humlicek's own test.
 *
 * kexp and ksincos are inlined as branch-free copies valid on region IV's
 * arguments (y^2 - x'^2 > -31; |2 x' y| < 11), with the same polynomials and
 * reductions: kexp's 2^k is built exactly from k + 1.5 * 2^52, ksincos's
 * quadrant is chosen by selects. Measured 3.0-3.3x faster with AVX2, 1.3x
 * without. */

static inline double URR_NAME(kexp)(double x) URR_TARGET;
static inline double URR_NAME(kexp)(double x) {
    double k=floor(x*1.44269504088896338700+.5);
    double r=(x-k*6.93147180369123816490e-01)-k*1.90821492927058770002e-10;
    double p=1.+r*(1.+r*(1./2+r*(1./6+r*(1./24+r*(1./120+r*(1./720+r*(1./5040
             +r*(1./40320+r*(1./362880+r*(1./3628800+r*(1./39916800+r*(1./479001600
             +r*(1./6227020800.)))))))))))));
    double t=k+6755399441055744.0; uint64_t u; memcpy(&u,&t,8);
    u=(u-0x4338000000000000ULL+1023ULL)<<52; double scale; memcpy(&scale,&u,8);
    return p*scale;
}

static inline void URR_NAME(ksincos)(double a,double *sine,double *cosine) URR_TARGET;
static inline void URR_NAME(ksincos)(double a,double *sine,double *cosine) {
    double k=floor(a*6.36619772367581382433e-01+.5);
    double r=(a-k*1.57079632673412561417e+00)-k*6.07710050650619224932e-11;
    double z=r*r;
    double sr=r+r*z*(-1.66666666666666324348e-01+z*(8.33333333332248946124e-03+z*(-1.98412698298579493134e-04
              +z*(2.75573137070700676789e-06+z*(-2.50507602534068634195e-08+z*1.58969099521155010221e-10)))));
    double cr=1.-.5*z+z*z*(4.16666666666666019037e-02+z*(-1.38888888888741095749e-03+z*(2.48015872894767294178e-05
              +z*(-2.75573143513906633035e-07+z*(2.08757232129817482790e-09+z*-1.13596475577881948265e-11)))));
    double q=k-4.*floor(k*0.25);
    double odd=q-2.*floor(q*0.5);
    double s=odd>0.5?cr:sr, c=odd>0.5?sr:cr;
    *sine=q>1.5?-s:s; *cosine=q>2.5?c:(q>0.5?-c:c);
}

/* Each region adds its run: el += cp Re w + sp Im w, ca += gp Re w, fi += fp Re w. */
#define URR_ACCUMULATE el[i]+=cp*wr+sp*wi;ca[i]+=gp*wr;fi[i]+=fp*wr;
#define URR_REGION_ARGS Py_ssize_t a,Py_ssize_t b,const double *restrict es,double er,double invdelta,double y, \
    double cp,double sp,double gp,double fp,double *restrict el,double *restrict ca,double *restrict fi

static void URR_NAME(region1)(URR_REGION_ARGS) URR_TARGET;
static void URR_NAME(region1)(URR_REGION_ARGS) {
    for (Py_ssize_t i=a;i<b;i++) {
        double x=(es[i]-er)*invdelta,tr=y,ti=-x;
        double ur=tr*tr-ti*ti+0.5,ui=2.*tr*ti;
        double nr=tr*0.5641896,ni=ti*0.5641896;
        double inv=1./(ur*ur+ui*ui);
        double wr=(nr*ur+ni*ui)*inv,wi=(ni*ur-nr*ui)*inv;
        URR_ACCUMULATE
    }
}

static void URR_NAME(region2)(URR_REGION_ARGS) URR_TARGET;
static void URR_NAME(region2)(URR_REGION_ARGS) {
    for (Py_ssize_t i=a;i<b;i++) {
        double x=(es[i]-er)*invdelta,tr=y,ti=-x;
        double ur=tr*tr-ti*ti,ui=2.*tr*ti;
        double pr=1.410474+ur*0.5641896,pi=ui*0.5641896;
        double nr=tr*pr-ti*pi,ni=tr*pi+ti*pr;
        double qr=3.+ur,qi=ui;
        double dr=ur*qr-ui*qi+0.75,di=ur*qi+ui*qr;
        double inv=1./(dr*dr+di*di);
        double wr=(nr*dr+ni*di)*inv,wi=(ni*dr-nr*di)*inv;
        URR_ACCUMULATE
    }
}

static void URR_NAME(region3)(URR_REGION_ARGS) URR_TARGET;
static void URR_NAME(region3)(URR_REGION_ARGS) {
    for (Py_ssize_t i=a;i<b;i++) {
        double x=(es[i]-er)*invdelta,tr=y,ti=-x,m;
        double nr=3.778987+tr*0.5642236,ni=ti*0.5642236;
        m=tr*nr-ti*ni;ni=tr*ni+ti*nr;nr=m+11.96482;
        m=tr*nr-ti*ni;ni=tr*ni+ti*nr;nr=m+20.20933;
        m=tr*nr-ti*ni;ni=tr*ni+ti*nr;nr=m+16.4955;
        double dr=6.699398+tr,di=ti;
        m=tr*dr-ti*di;di=tr*di+ti*dr;dr=m+21.69274;
        m=tr*dr-ti*di;di=tr*di+ti*dr;dr=m+39.27121;
        m=tr*dr-ti*di;di=tr*di+ti*dr;dr=m+38.82363;
        m=tr*dr-ti*di;di=tr*di+ti*dr;dr=m+16.4955;
        double inv=1./(dr*dr+di*di);
        double wr=(nr*dr+ni*di)*inv,wi=(ni*dr-nr*di)*inv;
        URR_ACCUMULATE
    }
}

/* Region IV: exp(t^2) minus a rational in u = t^2 (coefficients unrolled so the loop vectorizes). */
#define URR_HORNER(pr,pi,c) m=ur*pr-ui*pi;pi=ur*pi+ui*pr;pr=(c)-m;pi=-pi;
static void URR_NAME(region4)(URR_REGION_ARGS) URR_TARGET;
static void URR_NAME(region4)(URR_REGION_ARGS) {
    for (Py_ssize_t i=a;i<b;i++) {
        double x=(es[i]-er)*invdelta,tr=y,ti=-x,m;
        double ur=tr*tr-ti*ti,ui=2.*tr*ti;
        double nr=0.56419,ni=0.,dr=1.,di=0.;
        URR_HORNER(nr,ni,1.320522) URR_HORNER(nr,ni,35.76683) URR_HORNER(nr,ni,219.0313)
        URR_HORNER(nr,ni,1540.787) URR_HORNER(nr,ni,3321.9905) URR_HORNER(nr,ni,36183.31)
        URR_HORNER(dr,di,1.841439) URR_HORNER(dr,di,61.57037) URR_HORNER(dr,di,364.2191)
        URR_HORNER(dr,di,2186.181) URR_HORNER(dr,di,9022.228) URR_HORNER(dr,di,24322.84)
        URR_HORNER(dr,di,32066.6)
        double magnitude=URR_NAME(kexp)(ur),sine,cosine;URR_NAME(ksincos)(ui,&sine,&cosine);
        double tnr=tr*nr-ti*ni,tni=tr*ni+ti*nr;
        double inv=1./(dr*dr+di*di);
        double wr=magnitude*cosine-(tnr*dr+tni*di)*inv,wi=magnitude*sine-(tni*dr-tnr*di)*inv;
        URR_ACCUMULATE
    }
}
#undef URR_HORNER

static void URR_NAME(ladder)(Py_ssize_t ns,Py_ssize_t nr,double c2,double s2,double delta,double reachw,double reachd,
        const double *es,const double *er,const double *gt,const double *gn,const double *gg,const double *gf,
        const double *s0,double *el,double *ca,double *fi) URR_TARGET;
static void URR_NAME(ladder)(Py_ssize_t ns,Py_ssize_t nr,double c2,double s2,double delta,double reachw,double reachd,
        const double *es,const double *er,const double *gt,const double *gn,const double *gg,const double *gf,
        const double *s0,double *el,double *ca,double *fi) {
    const double rpi=1.7724538509055160273;
    double invdelta=delta>0?1./delta:0.;
    for (Py_ssize_t r=0;r<nr;r++) {
        if (!(gt[r]>0)) continue;
        double reach=reachw*gt[r]+fmax(reachw*gt[r],reachd*delta);
        double lo=er[r]-reach,hi=er[r]+reach;
        Py_ssize_t a=0,z=ns;
        while (a<z) {Py_ssize_t m=(a+z)/2;if (es[m]<lo) a=m+1; else z=m;}
        z=a;while (z<ns && es[z]<=hi) z++;
        double ratio=gn[r]/gt[r],cel=s0[r]*(c2-(1.-ratio)),sel=s0[r]*s2;
        double ccap=s0[r]*gg[r]/gt[r],cfis=s0[r]*gf[r]/gt[r];
        if (!(delta>0)) {
            double i2=2./gt[r];
            for (Py_ssize_t i=a;i<z;i++) {
                double x=(es[i]-er[r])*i2,psi=1./(1.+x*x),chi=x*psi;
                el[i]+=cel*psi+sel*chi;ca[i]+=ccap*psi;fi[i]+=cfis*psi;
            }
            continue;
        }
        double y=0.5*gt[r]*invdelta,k=rpi*y;   /* psi = sqrt(pi)/2 xi Re w, xi = 2y */
        double cp=cel*k,sp=sel*k,gp=ccap*k,fp=cfis*k,e0=er[r];
        Py_ssize_t i=a;
        while (i<z) {
            double x=(es[i]-e0)*invdelta,s=fabs(x)+y;
            int region=s>=15.?1:(s>=5.5?2:(y>=0.195*fabs(x)-0.176?3:4));
            Py_ssize_t j=i+1;
            for (;j<z;j++) {
                double xj=(es[j]-e0)*invdelta,sj=fabs(xj)+y;
                int rj=sj>=15.?1:(sj>=5.5?2:(y>=0.195*fabs(xj)-0.176?3:4));
                if (rj!=region) break;
            }
            if (region==1) URR_NAME(region1)(i,j,es,e0,invdelta,y,cp,sp,gp,fp,el,ca,fi);
            else if (region==2) URR_NAME(region2)(i,j,es,e0,invdelta,y,cp,sp,gp,fp,el,ca,fi);
            else if (region==3) URR_NAME(region3)(i,j,es,e0,invdelta,y,cp,sp,gp,fp,el,ca,fi);
            else URR_NAME(region4)(i,j,es,e0,invdelta,y,cp,sp,gp,fp,el,ca,fi);
            i=j;
        }
    }
}
#undef URR_ACCUMULATE
#undef URR_REGION_ARGS
