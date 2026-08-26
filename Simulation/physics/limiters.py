from numba import njit
import numpy as np

# Limiters
@njit
def minmod(r):
	return max(0.0, min(1.0, r))

@njit
def vanalbada(r):
    return 0.0 if r <= 0.0 else (r * r + r) / (r * r + 1.0)

@njit
def koren(r):
    return max(0.0, min(2.0 * r, min((1.0 + 2.0 * r) / 3.0, 2.0)))

@njit
def ospre(r):
	return 0.0 if r <= 0.0 else (1.5 * r * (r + 1)) / (r * r + r + 1.0)

@njit
def vanleer(r):
	return 0.0 if r <= 0.0 else 2.0 * r / (1.0 + r)

@njit
def mc(r):
    return max(0.0, min(min(2.0 * r, 0.5 * (1.0 + r)), 2.0))

@njit
def umist(r):
    return max(0.0, min(min(2.0 * r, 0.25 + 0.75 * r), min(0.75 + 0.25 * r, 2.0)))

@njit
def superbee(r):
	return max(0.0, max(min(1.0, 2.0 * r), min(2.0, r)))

@njit
def smart(r):
	return max(0.0, min(min(2.0 * r, 0.75 * r + 0.25), 4.0))

@njit
def charm(r):
    return 0.0 if r <= 0.0 else (r * (3.0 * r + 1.0)) / ((r + 1.0) * (r + 1.0))

@njit
def quick(r):
    return max(0.0, min(min(2.0 * r, (3.0 + r) / 4.0), 2.0))

@njit
def sweby(r, beta=1.5):
    # при beta = 1.0 це minmod, при beta = 2.0 це superbee
    if r <= 0.0:
        return 0.0
    return max(min(beta * r, 1.0), min(r, beta))

@njit
def vanalbada2(r):
    return 0.0 if r <= 0.0 else (2.0 * r) / (r * r + 1.0)

@njit
def hcus(r):
    return 0.0 if r <= 0.0 else (1.5 * (r + abs(r))) / (r + 2.0)

@njit
def hquick(r):
    """HQUICK (High-order QUICK) limiter"""
    return 0.0 if r <= 0.0 else (2.0 * (r + abs(r))) / (r + 3.0)

@njit
def lax_wendroff(r):
    return 1.0

@njit
def beam_warming(r):
    return max(0.0, r)

@njit
def fromm(r):
    return max(0.0, 0.5 * (1.0 + r))

@njit
def ultrabee(r):
    return max(0.0, min(2.0 * r, 2.0))

@njit
def generalized_minmod(r, theta=1.5):
    # При theta = 1.0 — Minmod, при theta = 2.0 — Superbee.
    if r <= 0.0:
        return 0.0
    return min(theta * r, min(0.5 * (1.0 + r), theta))

@njit
def gamma_scheme(r, beta=0.1):
    if r <= 0.0:
        return 0.0
    return min(2.0 * r / beta, min(1.0 + r, 2.0))

@njit
def venkatakrishnan_1d(r):
    if r <= 0.0:
        return 0.0
    return (r * r + 2.0 * r) / (r * r + r + 2.0)

@njit
def cui(r):
    return max(0.0, (3.0 + r) / 4.0)

@njit
def cubista(r):
    if r <= 0.0:
        return 0.0
    return min(2.0 * r, min(0.25 + 0.75 * r, 2.0))