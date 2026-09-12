#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration symplectique et chaos hamiltonien.
(A) Kepler : hamiltonien modifie du saute-mouton, verifie a l'ordre dt^4.
(B) Henon-Heiles : sections de Poincare, destruction des tores KAM.
(C) Exposants de Lyapunov par l'algorithme de Benettin.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================== (A) KEPLER
def kepler_acc(q):
    r = np.linalg.norm(q)
    return -q/r**3

def H_kepler(q, p):
    return 0.5*np.dot(p, p) - 1.0/np.linalg.norm(q)

def Hmod_kepler(q, p, dt):
    """H~ = H + dt^2 [ -(1/24)|grad V|^2 + (1/12) p_i p_j d_i d_j V ]"""
    r = np.linalg.norm(q); p2 = np.dot(p, p); pq = np.dot(p, q)
    gradV2 = 1.0/r**4
    ppHess = p2/r**3 - 3*pq**2/r**5
    return H_kepler(q, p) + dt**2*(-gradV2/24.0 + ppHess/12.0)

def leapfrog_kepler(q0, p0, dt, nsteps):
    q, p = q0.copy(), p0.copy()
    out = np.empty((nsteps+1, 5))
    out[0] = [0.0, H_kepler(q,p), Hmod_kepler(q,p,dt), q[0]*p[1]-q[1]*p[0], np.linalg.norm(q)]
    for n in range(nsteps):
        p = p + 0.5*dt*kepler_acc(q)
        q = q + dt*p
        p = p + 0.5*dt*kepler_acc(q)
        out[n+1] = [(n+1)*dt, H_kepler(q,p), Hmod_kepler(q,p,dt),
                    q[0]*p[1]-q[1]*p[0], np.linalg.norm(q)]
    return out

def rk4_kepler(q0, p0, dt, nsteps):
    def f(y):
        q, p = y[:2], y[2:]
        return np.concatenate([p, kepler_acc(q)])
    y = np.concatenate([q0, p0]); out = np.empty((nsteps+1, 2))
    out[0] = [0.0, H_kepler(y[:2], y[2:])]
    for n in range(nsteps):
        k1 = f(y); k2 = f(y+0.5*dt*k1); k3 = f(y+0.5*dt*k2); k4 = f(y+dt*k3)
        y = y + dt*(k1+2*k2+2*k3+k4)/6
        out[n+1] = [(n+1)*dt, H_kepler(y[:2], y[2:])]
    return out

# ============================================================ (B,C) HENON-HEILES
def hh_V(x, y):   return 0.5*(x*x+y*y) + x*x*y - y**3/3
def hh_gradV(x, y): return np.array([x + 2*x*y, y + x*x - y*y])
def hh_hess(x, y):  return np.array([[1+2*y, 2*x], [2*x, 1-2*y]])

def hh_leapfrog(q, p, dt, n):
    Q = np.empty((n+1,2)); P = np.empty((n+1,2))
    Q[0], P[0] = q, p
    for i in range(n):
        p = p - 0.5*dt*hh_gradV(*q)
        q = q + dt*p
        p = p - 0.5*dt*hh_gradV(*q)
        Q[i+1], P[i+1] = q, p
    return Q, P

def poincare_batch(E, y0s, py0s, dt=3e-3, T=2500.0):
    """Section x=0, px>0, vectorisee sur un lot de conditions initiales."""
    y0s = np.asarray(y0s, float); py0s = np.asarray(py0s, float)
    px2 = 2*(E - hh_V(0.0, y0s)) - py0s**2
    ok = px2 > 1e-12
    x = np.zeros(ok.sum()); y = y0s[ok].copy()
    px = np.sqrt(px2[ok]); py = py0s[ok].copy()
    n = int(T/dt)
    out = [[] for _ in range(len(x))]
    for i in range(n):
        gx = x + 2*x*y;  gy = y + x*x - y*y
        px -= 0.5*dt*gx; py -= 0.5*dt*gy
        xo = x.copy()
        x = x + dt*px;   y = y + dt*py
        gx = x + 2*x*y;  gy = y + x*x - y*y
        px -= 0.5*dt*gx; py -= 0.5*dt*gy
        hit = (xo < 0) & (x >= 0) & (px > 0)
        for j in np.nonzero(hit)[0]:
            f = -xo[j]/(x[j]-xo[j])
            out[j].append((y[j] - (1-f)*dt*py[j], py[j]))
    return [np.array(o) for o in out if len(o)]

def lyapunov_all(cases, dt=5e-3, T=12000.0, renorm=200):
    """cases = liste de (E, y0, py0). Benettin vectorise sur TOUS les cas."""
    E   = np.array([c[0] for c in cases]); y = np.array([c[1] for c in cases])
    py  = np.array([c[2] for c in cases]); x = np.zeros(len(cases))
    px2 = 2*(E - hh_V(0.0, y)) - py**2
    px  = np.sqrt(np.maximum(px2, 0.0))
    dx = np.ones(len(cases)); dy = dpx = dpy = np.zeros(len(cases))
    dy = np.zeros(len(cases)); dpx = np.zeros(len(cases)); dpy = np.zeros(len(cases))
    n = int(T/dt); acc = np.zeros(len(cases))
    for i in range(1, n+1):
        h11 = 1+2*y; h12 = 2*x; h22 = 1-2*y
        px -= 0.5*dt*(x + 2*x*y); py -= 0.5*dt*(y + x*x - y*y)
        dpx -= 0.5*dt*(h11*dx + h12*dy); dpy -= 0.5*dt*(h12*dx + h22*dy)
        x = x + dt*px; y = y + dt*py
        dx = dx + dt*dpx; dy = dy + dt*dpy
        h11 = 1+2*y; h12 = 2*x; h22 = 1-2*y
        px -= 0.5*dt*(x + 2*x*y); py -= 0.5*dt*(y + x*x - y*y)
        dpx -= 0.5*dt*(h11*dx + h12*dy); dpy -= 0.5*dt*(h12*dx + h22*dy)
        if i % renorm == 0:
            nr = np.sqrt(dx**2+dy**2+dpx**2+dpy**2)
            acc += np.log(nr); dx/=nr; dy/=nr; dpx/=nr; dpy/=nr
    return acc/(n*dt), px2 > 1e-12

# ==================================================================== MAIN
if __name__ == "__main__":
    print("="*74)
    print("(A) KEPLER : hamiltonien modifie du saute-mouton")
    print("="*74)
    e = 0.5
    q0 = np.array([1-e, 0.0]); p0 = np.array([0.0, np.sqrt((1+e)/(1-e))])
    E0 = H_kepler(q0, p0); L0 = q0[0]*p0[1]-q0[1]*p0[0]
    print(f"  orbite d'excentricite {e} : E0 = {E0:.10f}, L0 = {L0:.10f}")
    print(f"  periode T = 2 pi a^(3/2) = {2*np.pi:.6f} (a=1)")

    print(f"\n  derive d'energie sur 5000 periodes (dt = 0.05, meme cout) :")
    dt = 0.05; nP = 2500; nst = int(nP*2*np.pi/dt)
    lf = leapfrog_kepler(q0, p0, dt, nst)
    rk = rk4_kepler(q0, p0, dt, nst)
    dHlf = (lf[:,1]-E0)/abs(E0); dHrk = (rk[:,1]-E0)/abs(E0)
    tP = lf[:,0]/(2*np.pi)
    def envelope(t, d, nb=40):
        idx = np.array_split(np.arange(len(t)), nb)
        return np.array([t[i].mean() for i in idx]), np.array([np.abs(d[i]).max() for i in idx])
    te, ee_l = envelope(tP, dHlf); _, ee_r = envelope(tP, dHrk)
    m = te > 50
    sl = np.polyfit(np.log(te[m]), np.log(ee_l[m]), 1)[0]
    sr = np.polyfit(np.log(te[m]), np.log(ee_r[m]), 1)[0]
    print(f"    saute-mouton : |dH/H| max = {np.abs(dHlf).max():.3e}  "
          f"croissance de l'enveloppe : t^({sl:+.3f})")
    print(f"    RK4          : |dH/H| max = {np.abs(dHrk).max():.3e}  "
          f"croissance de l'enveloppe : t^({sr:+.3f})")
    print(f"    a t = {nP} periodes : saute-mouton {np.abs(dHlf[-1]):.3e}, "
          f"RK4 {np.abs(dHrk[-1]):.3e}")
    print(f"    -> l'erreur du saute-mouton est BORNEE, celle de RK4 croit "
          f"lineairement.")
    print(f"  moment cinetique (saute-mouton) : |dL| max = "
          f"{np.abs(lf[:,3]-L0).max():.3e}  (conserve exactement)")

    print(f"\n  verification du hamiltonien modifie : ordre en dt")
    print(f"{'dt':>8}{'max|dH|':>14}{'ordre':>8}{'max|dH~|':>14}{'ordre':>8}{'gain':>10}")
    prevH = prevM = None
    for dt in [0.04, 0.02, 0.01, 0.005]:
        n = int(20*2*np.pi/dt)
        o = leapfrog_kepler(q0, p0, dt, n)
        aH = np.abs(o[:,1]-o[0,1]).max()
        aM = np.abs(o[:,2]-o[0,2]).max()
        rH = np.log2(prevH/aH) if prevH else np.nan
        rM = np.log2(prevM/aM) if prevM else np.nan
        print(f"{dt:8.3f}{aH:14.3e}{rH:8.2f}{aM:14.3e}{rM:8.2f}{aH/aM:10.1f}")
        prevH, prevM = aH, aM
    print("  -> H varie en dt^2, H~ en dt^4 : le saute-mouton integre exactement")
    print("     un hamiltonien voisin, a dt^4 pres.")

    print()
    print("="*74)
    print("(B) HENON-HEILES : destruction des tores KAM")
    print("="*74)
    print("  H = (px^2+py^2)/2 + (x^2+y^2)/2 + x^2 y - y^3/3 ;  E_echappement = 1/6")
    Es = [1/12, 1/8, 1/6]
    secs = {}
    ys  = np.repeat(np.linspace(-0.35, 0.55, 11), 2)
    pys = np.tile([0.0, 0.12], 11)
    for E in Es:
        cur = poincare_batch(E, ys, pys, dt=4e-3, T=1800.0)
        secs[E] = cur
        npts = sum(len(c) for c in cur)
        print(f"   E = {E:.5f} ({E*6:.2f} x E_ech) : {len(cur)} orbites, {npts} points")

    print()
    print("="*74)
    print("(C) EXPOSANTS DE LYAPUNOV (Benettin)")
    print("="*74)
    print(f"{'E':>10}{'C.I. (y,py)':>18}{'lambda':>12}   nature")
    cases = [(1/12,0.0,0.0),(1/12,0.30,0.0),(1/12,-0.20,0.10),
             (1/8, 0.0,0.0),(1/8, 0.30,0.0),(1/8, -0.20,0.10),
             (1/6, 0.0,0.0),(1/6, 0.30,0.0),(1/6, -0.15,0.05)]
    lams, valid = lyapunov_all(cases)
    for (E,y0,py0), lam, v in zip(cases, lams, valid):
        if not v: continue
        nat = ("chaotique" if lam > 0.03 else
               "marginale (orbite collante)" if lam > 0.004 else
               "reguliere (tore KAM)")
        print(f"{E:10.5f}{f'({y0:+.2f},{py0:+.2f})':>18}{lam:12.5f}   {nat}")

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.3,
                         "figure.dpi":150,"savefig.bbox":"tight","axes.linewidth":.8})
    C1,C2,C3="#1f3b73","#b3452c","#2f7d4f"
    fig,axs=plt.subplots(1,2,figsize=(9.4,3.2))
    m = slice(0, len(lf), 91)
    axs[0].plot(lf[m,0]/(2*np.pi), dHlf[m], color=C1, lw=.8, label="saute-mouton (symplectique)")
    axs[0].plot(rk[m,0]/(2*np.pi), dHrk[m], color=C2, lw=1.2, label="RK4 (ordre 4, non symplectique)")
    axs[0].set_xlabel("periodes"); axs[0].set_ylabel(r"$\Delta H/|H_0|$")
    axs[0].legend(fontsize=7.5); axs[0].set_title("derive seculaire ou pas",fontsize=9.5)
    o = leapfrog_kepler(q0, p0, 0.02, int(20*2*np.pi/0.02))
    axs[1].semilogy(o[:,0]/(2*np.pi), np.abs(o[:,1]-o[0,1])+1e-18, color=C1, lw=.9,
                    label=r"$|\Delta H|$")
    axs[1].semilogy(o[:,0]/(2*np.pi), np.abs(o[:,2]-o[0,2])+1e-18, color=C3, lw=.9,
                    label=r"$|\Delta \tilde H|$ (modifie)")
    axs[1].set_xlabel("periodes"); axs[1].set_ylabel("erreur")
    axs[1].legend(fontsize=8); axs[1].set_title(r"le hamiltonien modifie, lui, est conserve",fontsize=9.5)
    fig.savefig("/home/claude/s_kepler.pdf")

    fig,axs=plt.subplots(1,3,figsize=(11.5,3.6))
    for ax,E in zip(axs,Es):
        for c in secs[E]:
            ax.plot(c[:,0],c[:,1],",",ms=.4,rasterized=True)
        ax.set_title(rf"$E={E:.4f}$  $({E*6:.2f}\,E_{{\rm ech}})$",fontsize=9)
        ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$p_y$"); ax.grid(alpha=.15)
    fig.savefig("/home/claude/s_poincare.pdf", dpi=170)
    print("\nfigures ecrites.")
