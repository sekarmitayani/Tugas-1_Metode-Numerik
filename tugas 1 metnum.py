import math
import numpy as np

# -----------------------------
# Sistem & Jacobian
# -----------------------------
def f1(x, y): return x**2 + x*y - 10
def f2(x, y): return y + 3*x*y**2 - 57

def F(x, y): return np.array([f1(x,y), f2(x,y)])
def J(x, y):
    return np.array([[2*x + y, x],
                     [3*y**2, 1 + 6*x*y]], float)

# -----------------------------
# Iterasi Seidel – Versi 1 (linier)
# -----------------------------
def g1_lin(x,y):
    return (10 - x**2)/y

def g2_lin(x_new,y):
    return 57 - 3*x_new*y**2

# -----------------------------
# Iterasi Seidel – Versi 2 (akar)
# -----------------------------
def g1_sqrt(x,y):
    val = 10 - x*y
    return math.sqrt(val) if val >= 0 else float("nan")

def g2_sqrt(x_new,y):
    if x_new == 0:
        return float("nan")
    val = (57 - y)/(3*x_new)
    return math.sqrt(val) if val >= 0 else float("nan")

# -----------------------------
# Iterasi Seidel umum
# -----------------------------
def seidel_iter(x0, y0, g1, g2, tol=1e-6, maxit=10, limit=None):
    """
    limit = None → tidak dibatasi (bisa divergen besar)
    limit = angka → batas nilai aman
    """
    x, y = x0, y0
    hist = [(0, x, y, 0.0, 0.0)]
    for i in range(1, maxit+1):
        xn = g1(x, y)
        yn = g2(xn, y)
        dx, dy = xn - x, yn - y
        hist.append((i, xn, yn, dx, dy))

        # deteksi divergensi
        if limit is not None:
            if not (math.isfinite(xn) and math.isfinite(yn)) or abs(xn) > limit or abs(yn) > limit:
                print(f"⚠️  Divergen (nilai > {limit:g}) pada iterasi {i}")
                return hist, False
        if max(abs(dx), abs(dy)) < tol:
            print(f"Konvergen dalam {i} iterasi → (x, y) ≈ ({xn:.6f}, {yn:.6f})")
            return hist, True
        x, y = xn, yn
    return hist, False

# -----------------------------
# Newton–Raphson
# -----------------------------
def newton(x0,y0,tol=1e-6,maxit=50):
    x, y = x0, y0
    hist=[(0,x,y,0.0,0.0,f1(x,y),f2(x,y))]
    for k in range(1,maxit+1):
        try:
            step = np.linalg.solve(J(x,y), -F(x,y))
        except np.linalg.LinAlgError:
            print("Jacobian singular pada iterasi", k)
            break
        xn, yn = x + step[0], y + step[1]
        dx, dy = abs(xn - x), abs(yn - y)
        hist.append((k, xn, yn, dx, dy, f1(xn,yn), f2(xn,yn)))
        if max(dx, dy) < tol:
            print(f"Newton konvergen dalam {k} iterasi → (x, y) ≈ ({xn:.6f}, {yn:.6f})")
            break
        x, y = xn, yn
    return hist

# -----------------------------
# Broyden (secant multivariabel)
# -----------------------------
def broyden(x0, y0, tol=1e-6, maxit=50):
    X = np.array([x0, y0])
    F0 = F(*X)
    h = 1e-6
    Fx, Fy = F(X[0]+h, X[1]), F(X[0], X[1]+h)
    B = np.column_stack(((Fx-F0)/h, (Fy-F0)/h))
    hist = [(0, X[0], X[1], 0.0, 0.0)]
    for k in range(1, maxit+1):
        try:
            s = np.linalg.solve(B, -F(*X))
        except np.linalg.LinAlgError:
            print("Aproksimasi Jacobian singular pada iterasi", k)
            break
        Xn = X + s
        dx, dy = abs(s[0]), abs(s[1])
        hist.append((k, Xn[0], Xn[1], dx, dy))
        if max(dx, dy) < tol:
            print(f"Broyden konvergen dalam {k} iterasi → (x, y) ≈ ({Xn[0]:.6f}, {Xn[1]:.6f})")
            break
        yv = F(*Xn) - F(*X)
        B = B + np.outer((yv - B @ s), s) / (s @ s)
        X = Xn
    return hist

# -----------------------------
# Cetak tabel
# -----------------------------
def print_table(title, hist, mode="IT"):
    print("\n" + "─"*65)
    print(f"⟦ {title} ⟧")
    print("─"*65)
    if mode == "IT":
        print(" i |        x         |        y         |    Δx     |    Δy")
        print("-"*75)
        for i, x, y, dx, dy in hist:
            print(f"{i:2d} | {x:12.6f} | {y:12.6f} | {dx:9.5f} | {dy:9.5f}")
    elif mode == "Newton":
        print(" i |        x         |        y         |    Δx     |    Δy     |      f1       |      f2")
        print("-"*100)
        for i, x, y, dx, dy, v1, v2 in hist:
            print(f"{i:2d} | {x:12.6f} | {y:12.6f} | {dx:9.5f} | {dy:9.5f} | {v1:10.3e} | {v2:10.3e}")
    print("─"*65)

# -----------------------------
# Main demo
# -----------------------------
if __name__ == "__main__":
    x0, y0 = 1.5, 3.5

    print("\n DEMONSTRASI M06 – Sistem Non-Linear (g1A & g2A)")
    print(f"Tebakan awal: x0={x0}, y0={y0}, toleransi=1e-6\n")

    # --- Seidel Versi 1: demo divergen penuh (tanpa limit)
    hist_lin_free, _ = seidel_iter(x0, y0, g1_lin, g2_lin, maxit=5, limit=None)
    print_table("Iterasi Seidel – Versi 1 (linier / divergen penuh)", hist_lin_free, "IT")

    # --- Seidel Versi 1: versi aman (dengan limit)
    hist_lin_safe, _ = seidel_iter(x0, y0, g1_lin, g2_lin, maxit=10, limit=1e6)
    print_table("Iterasi Seidel – Versi 1 (linier / divergen aman)", hist_lin_safe, "IT")

    # --- Seidel Versi 2 (akar kuadrat)
    hist_sqrt, _ = seidel_iter(x0, y0, g1_sqrt, g2_sqrt, maxit=10, limit=1e6)
    print_table("Iterasi Seidel – Versi 2 (akar / konvergen)", hist_sqrt, "IT")

    # --- Newton–Raphson
    hist_newton = newton(x0, y0)
    print_table("Newton–Raphson", hist_newton, "Newton")

    # --- Broyden
    hist_broyden = broyden(x0, y0)
    print_table("Broyden (secant multivariabel)", hist_broyden, "IT")

    # --- Ringkasan hasil
    print("\nRingkasan hasil akhir:")
    print(f"  Seidel (akar) : x = {hist_sqrt[-1][1]:.6f}, y = {hist_sqrt[-1][2]:.6f}")
    print(f"  Newton         : x = {hist_newton[-1][1]:.6f}, y = {hist_newton[-1][2]:.6f}")
    print(f"  Broyden        : x = {hist_broyden[-1][1]:.6f}, y = {hist_broyden[-1][2]:.6f}")
    print("───────────────────────────────────────────────")
