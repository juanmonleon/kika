/* Portable/AVX2 kernels, with ordinary and certified envelope paths. */
static RM_TARGET void RM_MATRICES(Py_ssize_t ne, Py_ssize_t n, Py_ssize_t c,
        const double *e,const double *er,const double *gamma,const double *a,
        const double *f,double *out,unsigned char *unsafe) {
    double max_f2[3]={0};int normal_energies=1;
    for (Py_ssize_t i=0;i<ne;i++)
        if (!(isfinite(e[i]) && fabs(e[i])<=1e20)) normal_energies=0;
    for (Py_ssize_t i=0;i<ne;i++) for (Py_ssize_t j=0;j<c;j++) {
        double value=f[i*c+j]*f[i*c+j];
        if (value>max_f2[j]) max_f2[j]=value;
    }
    for (Py_ssize_t start=0;start<ne;start+=TILE) {
        Py_ssize_t count=ne-start<TILE?ne-start:TILE;
        double rr[9*TILE]={0},ri[9*TILE]={0};
#ifndef RM_SKIP_ENVELOPE
        double envelope[3*TILE]={0};
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
