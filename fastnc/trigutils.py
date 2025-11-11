"""
Author: Sunao Sugiyama
Last edit  : 2024/03/25 22:02:25

This module provides the functions to manipulate the triangle configuration.
Triangle can be specified by several different ways, and this module provides
the conversion between them.

x1x2x3  (SSS) : Three side lengths parametrization. 
                Triangle orientation is ambiguous.
ruv     (RUV) : (r,u,v) parametrization introduced in M. Jarvis+2003
                Triangle orientation is expressed by sign of v.
x1x2phi (SAS) : Two side lengths and opening angle parametrization.
                Triangle orientation is expressed by sign of phi.
xpsimu        : Same as x1x2phi but ratio of x2/x1 is parametrized by tan(psi)
                and phi is reparametrized as mu=cos(phi).
"""

import numpy as np

def orientation_sign(x1, x2, x3):
    x = np.stack([x1, x2, x3], axis=-1)
    order = np.argsort(x, axis=-1)

    # (0,1,2), (1,2,0), (2,0,1) -> counterclockwise (+1)
    # (0,2,1), (2,1,0), (1,0,2) -> clockwise (-1)
    clockwise_set = np.array([[0,2,1],[2,1,0],[1,0,2]])

    # check if clockwise or not for each
    is_clockwise = np.any(np.all(order[..., None, :] == clockwise_set, axis=-1), axis=-1)

    sign = np.where(is_clockwise, -1, +1)
    return sign

# ruv <-> x1x2x3
def ruv_to_x1x2x3(r, u, v, signed=False):
    # x1 = r*(1+u*np.abs(v))
    if signed:
        x1 = r*(1+u*v)
    else:
        x1 = r*(1+u*np.abs(v))
    x2 = r
    x3 = r*u
    return x1, x2, x3

def x1x2x3_to_ruv(x1, x2, x3, signed=True, all_physical=True):
    """
    x1, x2, x3 are clockwise side lengths of triangle
    """
    # Get sorted side length
    d3, d2, d1 = np.sort([x1,x2,x3], axis=0)
    r = d2
    u = d3/d2
    v = (d1-d2)/d3

    if all_physical:
        u[u>1] = 1
        v[v<0] = 0
        v[v>1] = 1

    if signed:
        sign = orientation_sign(x1, x2, x3)
        v *= sign

    return r, u, v

# ruv <-> x1x2phi
def ruv_to_x1x2phi(r, u, v, rot=0):
    if rot == 0:
        x1, x2, x3 = (1+u*np.abs(v)), 1., u
    elif rot == 1:
        x2, x3, x1 = (1+u*np.abs(v)), 1., u
    elif rot == 2:
        x3, x1, x2 = (1+u*np.abs(v)), 1., u
    phi = np.arccos( (x1**2+x2**2-x3**2)/2/x1/x2 ) * (-np.sign(v))
    x1 *= r
    x2 *= r
    return x1, x2, phi

def x1x2phi_to_ruv(x1, x2, phi, rot=0):
    pass

# x1x2phi <-> x1x2x3
def x1x2phi_to_x1x2x3(x1, x2, phi):
    x3 = np.sqrt(x1**2 + x2**2 - 2*x1*x2*np.cos(phi))
    return x1, x2, x3

def x1x2x3_to_x1x2phi(x1, x2, x3):
    pass

# xpsimu <-> x1x2x3
def xpsimu_to_x1x2x3(x, psi, mu):
    x1 = x * np.cos(psi)
    x2 = x * np.sin(psi)
    x3 = x * np.sqrt(1 - np.sin(2*psi)*mu)
    return x1, x2, x3

def x1x2x3_to_xpsimu(x1, x2, x3):
    x = np.sqrt(x1**2 + x2**2)
    psi = np.arctan2(x2, x1)
    mu = (x1**2+x2**2-x3**2)/2/x1/x2
    return x, psi, mu