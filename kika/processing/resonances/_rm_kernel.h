/* Portable/AVX2 kernels, with ordinary and certified envelope paths.
 *
 * The level sums run over a block of RM_BLOCK energies with every
 * accumulator local to the block, levels innermost. Each energy still adds
 * its levels in order k = 0..n-1 with the same operations, so the sums are
 * bit-identical to a tile sweep; what changes is that the accumulators no
 * longer go through memory once per level (at three channels the old tile
 * sweep moved 12 x TILE doubles per level, about 1.4 ns per level-energy
 * pair). The number of channels is a compile-time constant in each body.
 *
 * The blocks need vector arithmetic (RM_VECTOR: GCC/clang, AVX2 target);
 * the baseline and MSVC builds keep the tile sweep, the same sums. */
#ifdef RM_VECTOR
#define RM_CAT2(x,y) x##y
#define RM_CAT(x,y) RM_CAT2(x,y)
#define RM_BODY RM_CAT(RM_MATRICES,_body)
#define RM_RANGE RM_CAT(RM_MATRICES,_range)
#define RM_JOB RM_CAT(RM_MATRICES,_job)
/* acc + x*y, rounded twice (no contraction). FMA was measured slower here
 * (1.62 against 1.22 ns per level-energy pair at three channels: the
 * accumulation chains are latency-bound) and gave up bit-identity. */
#define RM_MADD(x,y,acc) ((acc)+(x)*(y))

/* Rows [lo, hi) of the level sums. Rows are independent, so any split of
 * them over threads gives the numbers of one sweep. */
RM_INLINE RM_TARGET void RM_BODY(Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t n, const Py_ssize_t c,
        const double *e,const double *er,const double *gamma,const double *a,
        const double *f,double *out,unsigned char *unsafe,const unsigned char *flags) {
    for (Py_ssize_t start=lo;start<hi;start+=RM_BLOCK) {
        Py_ssize_t count=hi-start<RM_BLOCK?hi-start:RM_BLOCK;
        /* A short last block repeats its first energy in the spare lanes,
         * which are computed and never stored. */
        Py_ssize_t row[RM_BLOCK];double ev[RM_BLOCK];unsigned char bad[RM_BLOCK];
        for (int b=0;b<RM_BLOCK;b++) {row[b]=start+(b<count?b:0);ev[b]=e[row[b]];bad[b]=0;}
        rm_v4d vr[RM_MAX_CHANNELS*(RM_MAX_CHANNELS+1)/2],vi[RM_MAX_CHANNELS*(RM_MAX_CHANNELS+1)/2];
        rm_v4d ve={ev[0],ev[1],ev[2],ev[3]},zero={0.,0.,0.,0.},one={1.,1.,1.,1.};
        for (int p=0;p<c*(c+1)/2;p++) vr[p]=vi[p]=zero;
#ifndef RM_SKIP_ENVELOPE
        const rm_v4i magnitude={0x7fffffffffffffffLL,0x7fffffffffffffffLL,0x7fffffffffffffffLL,0x7fffffffffffffffLL};
        rm_v4d venv[RM_MAX_CHANNELS];
        for (Py_ssize_t j=0;j<c;j++) venv[j]=zero;
#endif
        for (Py_ssize_t k=0;k<n;k++) {
            double half=.5*gamma[k];rm_v4d qr,qi;
            if (flags[k]&1) {
                rm_v4d dr=er[k]-ve,inverse=one/(dr*dr+half*half);
                qr=dr*inverse;qi=half*inverse;
            } else {
                double sr[RM_BLOCK],si[RM_BLOCK];
                for (int b=0;b<RM_BLOCK;b++) {
                    double dr=er[k]-ev[b],den=dr*dr+half*half;
                    if (!(den>=DBL_MIN && isfinite(den))) {
                        bad[b]=1;sr[b]=si[b]=0.;continue;
                    }
                    double inverse=1./den;
                    sr[b]=dr*inverse;si[b]=half*inverse;
                    if (flags[k]&2) {
                        double strength=0.;
                        for (Py_ssize_t j=0;j<c;j++) {
                            double value=a[k*c+j]*f[row[b]*c+j];strength+=value*value;
                        }
                        double cut=1e-6*strength;
                        if (fabs(dr)<=cut && hypot(dr,half)<=cut) bad[b]=1;
                    }
                }
                qr=(rm_v4d){sr[0],sr[1],sr[2],sr[3]};qi=(rm_v4d){si[0],si[1],si[2],si[3]};
            }
#ifndef RM_SKIP_ENVELOPE
            rm_v4d size=(rm_v4d)((rm_v4i)qr&magnitude)+(rm_v4d)((rm_v4i)qi&magnitude);
#endif
            for (Py_ssize_t j=0,p=0;j<c;j++) {
#ifndef RM_SKIP_ENVELOPE
                double square=a[k*c+j]*a[k*c+j];
                rm_v4d sv={square,square,square,square};
                venv[j]=RM_MADD(sv,size,venv[j]);
#endif
                for (Py_ssize_t l=j;l<c;l++,p++) {
                    double product=a[k*c+j]*a[k*c+l];
                    rm_v4d pv={product,product,product,product};
                    vr[p]=RM_MADD(pv,qr,vr[p]);vi[p]=RM_MADD(pv,qi,vi[p]);
                }
            }
        }
        for (Py_ssize_t b=0;b<count;b++) {
            Py_ssize_t i=start+b;
            unsafe[i]=bad[b];
#ifndef RM_SKIP_ENVELOPE
            double total=0.;
            for (Py_ssize_t j=0;j<c;j++) total+=venv[j][b]*f[i*c+j]*f[i*c+j];
            if (!(isfinite(total) && total<=ENVELOPE_LIMIT)) unsafe[i]=1;
#endif
            for (Py_ssize_t j=0,p=0;j<c;j++) for (Py_ssize_t l=j;l<c;l++,p++) {
                double scale=f[i*c+j]*f[i*c+l];
                double real=vr[p][b]*scale,imag=vi[p][b]*scale;
                if (!(isfinite(real) && isfinite(imag))) unsafe[i]=1;
                out[2*(i*c*c+j*c+l)]=out[2*(i*c*c+l*c+j)]=real;
                out[2*(i*c*c+j*c+l)+1]=out[2*(i*c*c+l*c+j)+1]=imag;
            }
        }
    }
}

typedef struct {
    Py_ssize_t n,c;const double *e,*er,*gamma,*a,*f;double *out;unsigned char *unsafe;const unsigned char *flags;
} RM_JOB;

static RM_TARGET void RM_RANGE(void *context,Py_ssize_t lo,Py_ssize_t hi) {
    const RM_JOB *j=(const RM_JOB *)context;
    switch (j->c) {
    case 1: RM_BODY(lo,hi,j->n,1,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    case 2: RM_BODY(lo,hi,j->n,2,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    case 3: RM_BODY(lo,hi,j->n,3,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    case 4: RM_BODY(lo,hi,j->n,4,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    case 5: RM_BODY(lo,hi,j->n,5,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    default: RM_BODY(lo,hi,j->n,6,j->e,j->er,j->gamma,j->a,j->f,j->out,j->unsafe,j->flags);break;
    }
}

/* flags: n bytes of scratch from the caller, written here. threads: rows
 * split among that many threads (rm_parallel), the same numbers. */
static RM_TARGET void RM_MATRICES(Py_ssize_t ne, Py_ssize_t n, Py_ssize_t c,
        const double *e,const double *er,const double *gamma,const double *a,
        const double *f,double *out,unsigned char *unsafe,unsigned char *flags,int threads) {
    double max_f2[RM_MAX_CHANNELS]={0};int normal_energies=1;
    for (Py_ssize_t i=0;i<ne;i++)
        if (!(isfinite(e[i]) && fabs(e[i])<=1e20)) normal_energies=0;
    for (Py_ssize_t i=0;i<ne;i++) for (Py_ssize_t j=0;j<c;j++) {
        double value=f[i*c+j]*f[i*c+j];
        if (value>max_f2[j]) max_f2[j]=value;
    }
    /* Per level, once per call: bit 0 the direct path, bit 1 a pole
     * candidate whose rows are tested against the level strength. They
     * depend on every row (max_f2), so they are made before any split. */
    for (Py_ssize_t k=0;k<n;k++) {
        double half=.5*gamma[k],upper=0.;
        for (Py_ssize_t j=0;j<c;j++) upper+=a[k*c+j]*a[k*c+j]*max_f2[j];
        upper=nextafter(upper*(1+16*c*DBL_EPSILON),INFINITY);
        int candidate=half<=1e-6*upper || gamma[k]==0.;
        int direct=!candidate && normal_energies && fabs(er[k])<=1e20 &&
                   half>=5e-101 && half<=5e19;
        flags[k]=(unsigned char)(direct|(candidate<<1));
    }
    RM_JOB job={n,c,e,er,gamma,a,f,out,unsafe,flags};
    rm_parallel(ne,threads,RM_RANGE,&job);
}
#undef RM_BODY
#undef RM_RANGE
#undef RM_JOB

#else
static RM_TARGET void RM_MATRICES(Py_ssize_t ne, Py_ssize_t n, Py_ssize_t c,
        const double *e,const double *er,const double *gamma,const double *a,
        const double *f,double *out,unsigned char *unsafe,unsigned char *flags,int threads) {
    (void)flags;(void)threads;
    double max_f2[RM_MAX_CHANNELS]={0};int normal_energies=1;
    for (Py_ssize_t i=0;i<ne;i++)
        if (!(isfinite(e[i]) && fabs(e[i])<=1e20)) normal_energies=0;
    for (Py_ssize_t i=0;i<ne;i++) for (Py_ssize_t j=0;j<c;j++) {
        double value=f[i*c+j]*f[i*c+j];
        if (value>max_f2[j]) max_f2[j]=value;
    }
    for (Py_ssize_t start=0;start<ne;start+=TILE) {
        Py_ssize_t count=ne-start<TILE?ne-start:TILE;
        double rr[RM_MAX_CHANNELS*RM_MAX_CHANNELS*TILE]={0},ri[RM_MAX_CHANNELS*RM_MAX_CHANNELS*TILE]={0};
#ifndef RM_SKIP_ENVELOPE
        double envelope[RM_MAX_CHANNELS*TILE]={0};
#endif
        double qr[TILE],qi[TILE];
        for (Py_ssize_t i=0;i<count;i++) unsafe[start+i]=0;
        for (Py_ssize_t k=0;k<n;k++) {
            double half=.5*gamma[k],upper=0.;
            for (Py_ssize_t j=0;j<c;j++) upper+=a[k*c+j]*a[k*c+j]*max_f2[j];
            upper=nextafter(upper*(1+16*c*DBL_EPSILON),INFINITY);
            int candidate=half<=1e-6*upper || gamma[k]==0.;
            if (!candidate && normal_energies && fabs(er[k])<=1e20 &&
                    half>=5e-101 && half<=5e19) {
                /* 2e-201 <= denominator <= 5e40: no underflow,
                 * overflow or naked pole. The level order is unchanged. */
                for (Py_ssize_t i=0;i<count;i++) {
                    double dr=er[k]-e[start+i],inverse=1./(dr*dr+half*half);
                    qr[i]=dr*inverse;qi[i]=half*inverse;
                }
            } else for (Py_ssize_t i=0;i<count;i++) {
                double dr=er[k]-e[start+i],den=dr*dr+half*half;
                if (!(den>=DBL_MIN && isfinite(den))) {
                    unsafe[start+i]=1;qr[i]=qi[i]=0.;continue;
                }
                double inverse=1./den;
                qr[i]=dr*inverse;qi[i]=half*inverse;
                if (candidate) {
                    double strength=0.;
                    for (Py_ssize_t j=0;j<c;j++) {
                        double value=a[k*c+j]*f[(start+i)*c+j];strength+=value*value;
                    }
                    double cut=1e-6*strength;
                    if (fabs(dr)<=cut && hypot(dr,half)<=cut) unsafe[start+i]=1;
                }
            }
            for (Py_ssize_t j=0;j<c;j++) {
#ifndef RM_SKIP_ENVELOPE
                double square=a[k*c+j]*a[k*c+j];
                for (Py_ssize_t i=0;i<count;i++)
                    envelope[j*TILE+i]+=square*(fabs(qr[i])+fabs(qi[i]));
#endif
                for (Py_ssize_t l=j;l<c;l++) {
                    double product=a[k*c+j]*a[k*c+l];Py_ssize_t base=(j*c+l)*TILE;
                    for (Py_ssize_t i=0;i<count;i++) {
                        rr[base+i]+=product*qr[i];ri[base+i]+=product*qi[i];
                    }
                }
            }
        }
        for (Py_ssize_t i=0;i<count;i++) {
#ifndef RM_SKIP_ENVELOPE
            double total=0.;
            for (Py_ssize_t j=0;j<c;j++) total+=envelope[j*TILE+i]*f[(start+i)*c+j]*f[(start+i)*c+j];
            if (!(isfinite(total) && total<=ENVELOPE_LIMIT)) unsafe[start+i]=1;
#endif
            for (Py_ssize_t j=0;j<c;j++) for (Py_ssize_t l=j;l<c;l++) {
                double scale=f[(start+i)*c+j]*f[(start+i)*c+l];
                Py_ssize_t base=(j*c+l)*TILE+i;
                double real=rr[base]*scale,imag=ri[base]*scale;
                if (!(isfinite(real) && isfinite(imag))) unsafe[start+i]=1;
                out[2*((start+i)*c*c+j*c+l)]=out[2*((start+i)*c*c+l*c+j)]=real;
                out[2*((start+i)*c*c+j*c+l)+1]=out[2*((start+i)*c*c+l*c+j)+1]=imag;
            }
        }
    }
}

#endif

#ifndef RM_SKIP_ABSORPTION
static RM_TARGET void RM_ABSORPTION(Py_ssize_t ne,Py_ssize_t n,Py_ssize_t c,
        const double *e,const double *er,const double *gamma,const double *a,
        const double *f,const double *y,double *out) {
    for (Py_ssize_t start=0;start<ne;start+=TILE) {
        Py_ssize_t count=ne-start<TILE?ne-start:TILE;
        double sum[TILE]={0},correction[TILE]={0};
        double qr[TILE],qi[TILE],nr[TILE],ni[TILE];
        for (Py_ssize_t k=0;k<n;k++) {
            if (gamma[k]==0.) continue;
            double half=.5*gamma[k];
            for (Py_ssize_t i=0;i<count;i++) {
                double dr=er[k]-e[start+i],den=dr*dr+half*half;
                if (!(den>=DBL_MIN && isfinite(den))) {
                    sum[i]=NAN;qr[i]=qi[i]=0.;
                } else {
                    double inverse=1./den;qr[i]=dr*inverse;qi[i]=half*inverse;
                }
                nr[i]=ni[i]=0.;
            }
            for (Py_ssize_t j=0;j<c;j++) {
                double amplitude=a[k*c+j];
                for (Py_ssize_t i=0;i<count;i++) {
                    Py_ssize_t at=(start+i)*c+j;
                    nr[i]+=amplitude*f[at]*y[2*at];ni[i]+=amplitude*f[at]*y[2*at+1];
                }
            }
            for (Py_ssize_t i=0;i<count;i++) {
                double xr=nr[i]*qr[i]-ni[i]*qi[i],xi=nr[i]*qi[i]+ni[i]*qr[i];
                double value=gamma[k]*(xr*xr+xi*xi);
                double next=value-correction[i],total=sum[i]+next;
                correction[i]=(total-sum[i])-next;sum[i]=total;
            }
        }
        for (Py_ssize_t i=0;i<count;i++) out[start+i]=2*sum[i];
    }
}


#endif
