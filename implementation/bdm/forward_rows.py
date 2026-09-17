"""Eight independent row registers with an explicit FP32 reduction tree."""
import triton as tr
import triton.language as tl

@tr.jit
def _sum8(x0,x1,x2,x3,x4,x5,x6,x7):
    return ((x0+x1)+(x4+x5))+((x2+x3)+(x6+x7))

@tr.jit
def _forward_rows(Q,K,V,G,B,W,INIT,CU,COFF,SNAP,OUT,ORDER,
                 WRITE_EVENTS:tl.constexpr,D:tl.constexpr,VD:tl.constexpr,
                 C:tl.constexpr,SCALE:tl.constexpr,INIT_SEQS:tl.constexpr,
                 STORE_OUTPUT:tl.constexpr=True,STORE_CHECKPOINTS:tl.constexpr=True,
                 SOURCE_V_KW:tl.constexpr=0,SOURCE_W_KW:tl.constexpr=0,WW=None,
                 FLAT_GRID:tl.constexpr=False, PRECOMPUTED_DECAY:tl.constexpr=False,
                 STORE_FINAL:tl.constexpr=False, READ_OUTPUT_ONLY:tl.constexpr=False,
                 ROUND_STATE:tl.constexpr=False):
    if FLAT_GRID:
        tile=tl.program_id(0)%tl.cdiv(D,VD)
        seq=(tl.program_id(0)//tl.cdiv(D,VD)).to(tl.int64)
    else:
        tile,seq=tl.program_id(0),tl.program_id(1)
    values=tile*VD+tl.arange(0,VD)
    start,end=tl.load(CU+seq),tl.load(CU+seq+1)
    if STORE_CHECKPOINTS:
        first=tl.load(COFF+seq)
    h0=tl.load(INIT+(seq%INIT_SEQS)*8*D+0*D+values,values<D,0).to(tl.float32)
    h1=tl.load(INIT+(seq%INIT_SEQS)*8*D+1*D+values,values<D,0).to(tl.float32)
    h2=tl.load(INIT+(seq%INIT_SEQS)*8*D+2*D+values,values<D,0).to(tl.float32)
    h3=tl.load(INIT+(seq%INIT_SEQS)*8*D+3*D+values,values<D,0).to(tl.float32)
    h4=tl.load(INIT+(seq%INIT_SEQS)*8*D+4*D+values,values<D,0).to(tl.float32)
    h5=tl.load(INIT+(seq%INIT_SEQS)*8*D+5*D+values,values<D,0).to(tl.float32)
    h6=tl.load(INIT+(seq%INIT_SEQS)*8*D+6*D+values,values<D,0).to(tl.float32)
    h7=tl.load(INIT+(seq%INIT_SEQS)*8*D+7*D+values,values<D,0).to(tl.float32)
    for e in range(start,end):
        if ROUND_STATE and (e-start)%C==0 and e>start:
            h0=h0.to(tl.bfloat16).to(tl.float32)
            h1=h1.to(tl.bfloat16).to(tl.float32)
            h2=h2.to(tl.bfloat16).to(tl.float32)
            h3=h3.to(tl.bfloat16).to(tl.float32)
            h4=h4.to(tl.bfloat16).to(tl.float32)
            h5=h5.to(tl.bfloat16).to(tl.float32)
            h6=h6.to(tl.bfloat16).to(tl.float32)
            h7=h7.to(tl.bfloat16).to(tl.float32)
        if STORE_CHECKPOINTS and (e-start)%C==0 and e>start:
            chunk=first+(e-start)//C-1
            tl.store(SNAP+chunk*8*D+0*D+values,h0,values<D)
            tl.store(SNAP+chunk*8*D+1*D+values,h1,values<D)
            tl.store(SNAP+chunk*8*D+2*D+values,h2,values<D)
            tl.store(SNAP+chunk*8*D+3*D+values,h3,values<D)
            tl.store(SNAP+chunk*8*D+4*D+values,h4,values<D)
            tl.store(SNAP+chunk*8*D+5*D+values,h5,values<D)
            tl.store(SNAP+chunk*8*D+6*D+values,h6,values<D)
            tl.store(SNAP+chunk*8*D+7*D+values,h7,values<D)
        if tl.load(ORDER+e)<WRITE_EVENTS:
            vi=tl.load(ORDER+e)//SOURCE_V_KW if SOURCE_V_KW else e
            v=tl.load(V+vi*D+values,values<D,0).to(tl.float32)
            if SOURCE_W_KW:
                original=tl.load(ORDER+e)
                w=tl.load(W+(original//SOURCE_W_KW)*D+values,values<D,0).to(tl.float32)
                w=(w*tl.load(WW+original).to(tl.float32)).to(W.dtype.element_ty).to(tl.float32)
            else:
                w=tl.load(W+e*D+values,values<D,0).to(tl.float32)
            k0=tl.load(K+e*8+0).to(tl.float32)
            g0=tl.load(G+e*8+0).to(tl.float32)
            b0=tl.load(B+e*8+0).to(tl.float32)
            hd0=h0*(g0 if PRECOMPUTED_DECAY else tl.exp(g0))
            t0=(b0*k0)*hd0
            k1=tl.load(K+e*8+1).to(tl.float32)
            g1=tl.load(G+e*8+1).to(tl.float32)
            b1=tl.load(B+e*8+1).to(tl.float32)
            hd1=h1*(g1 if PRECOMPUTED_DECAY else tl.exp(g1))
            t1=(b1*k1)*hd1
            k2=tl.load(K+e*8+2).to(tl.float32)
            g2=tl.load(G+e*8+2).to(tl.float32)
            b2=tl.load(B+e*8+2).to(tl.float32)
            hd2=h2*(g2 if PRECOMPUTED_DECAY else tl.exp(g2))
            t2=(b2*k2)*hd2
            k3=tl.load(K+e*8+3).to(tl.float32)
            g3=tl.load(G+e*8+3).to(tl.float32)
            b3=tl.load(B+e*8+3).to(tl.float32)
            hd3=h3*(g3 if PRECOMPUTED_DECAY else tl.exp(g3))
            t3=(b3*k3)*hd3
            k4=tl.load(K+e*8+4).to(tl.float32)
            g4=tl.load(G+e*8+4).to(tl.float32)
            b4=tl.load(B+e*8+4).to(tl.float32)
            hd4=h4*(g4 if PRECOMPUTED_DECAY else tl.exp(g4))
            t4=(b4*k4)*hd4
            k5=tl.load(K+e*8+5).to(tl.float32)
            g5=tl.load(G+e*8+5).to(tl.float32)
            b5=tl.load(B+e*8+5).to(tl.float32)
            hd5=h5*(g5 if PRECOMPUTED_DECAY else tl.exp(g5))
            t5=(b5*k5)*hd5
            k6=tl.load(K+e*8+6).to(tl.float32)
            g6=tl.load(G+e*8+6).to(tl.float32)
            b6=tl.load(B+e*8+6).to(tl.float32)
            hd6=h6*(g6 if PRECOMPUTED_DECAY else tl.exp(g6))
            t6=(b6*k6)*hd6
            k7=tl.load(K+e*8+7).to(tl.float32)
            g7=tl.load(G+e*8+7).to(tl.float32)
            b7=tl.load(B+e*8+7).to(tl.float32)
            hd7=h7*(g7 if PRECOMPUTED_DECAY else tl.exp(g7))
            t7=(b7*k7)*hd7
            erase=_sum8(t0,t1,t2,t3,t4,t5,t6,t7)
            u=w*v-erase
            h0=hd0+k0*u
            h1=hd1+k1*u
            h2=hd2+k2*u
            h3=hd3+k3*u
            h4=hd4+k4*u
            h5=hd5+k5*u
            h6=hd6+k6*u
            h7=hd7+k7*u
            result=tl.full((VD,),0,tl.float32)
        else:
            q0=tl.load(Q+e*8+0).to(tl.float32)
            t0=q0*h0
            q1=tl.load(Q+e*8+1).to(tl.float32)
            t1=q1*h1
            q2=tl.load(Q+e*8+2).to(tl.float32)
            t2=q2*h2
            q3=tl.load(Q+e*8+3).to(tl.float32)
            t3=q3*h3
            q4=tl.load(Q+e*8+4).to(tl.float32)
            t4=q4*h4
            q5=tl.load(Q+e*8+5).to(tl.float32)
            t5=q5*h5
            q6=tl.load(Q+e*8+6).to(tl.float32)
            t6=q6*h6
            q7=tl.load(Q+e*8+7).to(tl.float32)
            t7=q7*h7
            result=_sum8(t0,t1,t2,t3,t4,t5,t6,t7)*SCALE
        if STORE_OUTPUT:
            if READ_OUTPUT_ONLY:
                original=tl.load(ORDER+e).to(tl.int64)
                if original>=WRITE_EVENTS:
                    tl.store(OUT+(original-WRITE_EVENTS)*D+values,result,values<D)
            else:
                tl.store(OUT+e*D+values,result,values<D)

    if STORE_FINAL:
        tl.store(INIT+seq*8*D+0*D+values,h0,values<D)
        tl.store(INIT+seq*8*D+1*D+values,h1,values<D)
        tl.store(INIT+seq*8*D+2*D+values,h2,values<D)
        tl.store(INIT+seq*8*D+3*D+values,h3,values<D)
        tl.store(INIT+seq*8*D+4*D+values,h4,values<D)
        tl.store(INIT+seq*8*D+5*D+values,h5,values<D)
        tl.store(INIT+seq*8*D+6*D+values,h6,values<D)
        tl.store(INIT+seq*8*D+7*D+values,h7,values<D)


@tr.jit
def forward_rows(Q,K,V,G,B,W,INIT,CU,COFF,SNAP,OUT,ORDER,
                 WRITE_EVENTS:tl.constexpr,D:tl.constexpr,VD:tl.constexpr,
                 C:tl.constexpr,SCALE:tl.constexpr,INIT_SEQS:tl.constexpr,
                 STORE_OUTPUT:tl.constexpr=True,STORE_CHECKPOINTS:tl.constexpr=True,
                 SOURCE_V_KW:tl.constexpr=0,SOURCE_W_KW:tl.constexpr=0,WW=None,
                 FLAT_GRID:tl.constexpr=False):
    _forward_rows(Q,K,V,G,B,W,INIT,CU,COFF,SNAP,OUT,ORDER,
                  WRITE_EVENTS,D,VD,C,SCALE,INIT_SEQS,STORE_OUTPUT,STORE_CHECKPOINTS,
                  SOURCE_V_KW,SOURCE_W_KW,WW,FLAT_GRID,False)


@tr.jit
def forward_rows_precomputed(Q,K,V,G,B,W,INIT,CU,COFF,SNAP,OUT,ORDER,
                 WRITE_EVENTS:tl.constexpr,D:tl.constexpr,VD:tl.constexpr,
                 C:tl.constexpr,SCALE:tl.constexpr,INIT_SEQS:tl.constexpr,
                 STORE_OUTPUT:tl.constexpr=True,STORE_CHECKPOINTS:tl.constexpr=True,
                 SOURCE_V_KW:tl.constexpr=0,SOURCE_W_KW:tl.constexpr=0,WW=None,
                 FLAT_GRID:tl.constexpr=False,ROUND_STATE:tl.constexpr=False):
    _forward_rows(Q,K,V,G,B,W,INIT,CU,COFF,SNAP,OUT,ORDER,
                  WRITE_EVENTS,D,VD,C,SCALE,INIT_SEQS,STORE_OUTPUT,STORE_CHECKPOINTS,
                  SOURCE_V_KW,SOURCE_W_KW,WW,FLAT_GRID,True,False,False,ROUND_STATE)


@tr.jit
def forward_rows_stateful(Q,K,V,G,B,W,STATE,CU,OUT,ORDER,WW,
                          WRITE_EVENTS:tl.constexpr,D:tl.constexpr,VD:tl.constexpr,
                          SCALE:tl.constexpr,SEQS:tl.constexpr,KW:tl.constexpr,
                          FLAT_GRID:tl.constexpr):
    # Reuse the accepted forward recurrence; only retain its final registers.
    _forward_rows(Q,K,V,G,B,W,STATE,CU,CU,STATE,OUT,ORDER,
                  WRITE_EVENTS,D,VD,64,SCALE,SEQS,True,False,
                  KW,KW,WW,FLAT_GRID,True,True)


@tr.jit
def forward_rows_readonly_stateful(Q,K,V,G,B,W,STATE,CU,OUT,ORDER,WW,
                                   WRITE_EVENTS:tl.constexpr,D:tl.constexpr,VD:tl.constexpr,
                                   SCALE:tl.constexpr,SEQS:tl.constexpr,KW:tl.constexpr,
                                   FLAT_GRID:tl.constexpr):
    _forward_rows(Q,K,V,G,B,W,STATE,CU,CU,STATE,OUT,ORDER,
                  WRITE_EVENTS,D,VD,64,SCALE,SEQS,True,False,
                  KW,KW,WW,FLAT_GRID,True,True,True)
