import numpy as np


def triangle_edges(V, F):
    """
    Calculate triangle edge vectors and lengths.
    Returns:
        e0, e1, e2: (M,3) Edge vectors
        a,  b,  c : (M,)  Edge lengths, where:
        a = ||e0||, b = ||e1||, c = ||e2||
    Convention: Opposite vertices:
        e0 = v2 - v1 (corresponding to length a)
        e1 = v0 - v2 (corresponding to length b)
        e2 = v1 - v0 (corresponding to length c)
    """
    v0 = V[F[:,0]]
    v1 = V[F[:,1]]
    v2 = V[F[:,2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    a = np.linalg.norm(e0, axis=1)
    b = np.linalg.norm(e1, axis=1)
    c = np.linalg.norm(e2, axis=1)
    return e0, e1, e2, a, b, c

def triangle_area_from_edges(e2, e0):
    """
    Calculate the area of a triangle using the cross product of two edges:
      area = 0.5 * || e2 x e0 ||
    where:
        e2 = v1 - v0, e0 = v2 - v1
    """
    cross = np.cross(e2, e0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return area

def triangle_angles(a, b, c):
    """
    Calculate the three internal angles (radians → degrees) using the law of cosines,
    and clip values to avoid floating-point errors.
    Angle A corresponds to side a (opposite v0), and so on.
    """
    # 余弦值裁剪到 [-1,1]
    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))

    # 余弦定理：
    # cos(A) = (b^2 + c^2 - a^2) / (2bc)
    cosA = (b*b + c*c - a*a) / (2.0 * b * c + 1e-15)
    cosB = (c*c + a*a - b*b) / (2.0 * c * a + 1e-15)
    cosC = (a*a + b*b - c*c) / (2.0 * a * b + 1e-15)

    A = np.degrees(safe_acos(cosA))
    B = np.degrees(safe_acos(cosB))
    C = np.degrees(safe_acos(cosC))
    return A, B, C

def triangle_inradius(area, a, b, c):
    """
    内切圆半径 r = Area / s, 其中 s 为半周长 (a+b+c)/2
    """
    s = 0.5 * (a + b + c)
    r = np.where(s > 0, area / s, 0.0)
    return r

def triangle_circumradius(area, a, b, c):
    """
    外接圆半径 R = (a*b*c) / (4*Area)
    对面积很小的三角形做保护。
    """
    R = np.zeros_like(area)
    mask = area > 1e-20
    R[mask] = (a[mask] * b[mask] * c[mask]) / (4.0 * area[mask])
    R[~mask] = np.inf
    return R
