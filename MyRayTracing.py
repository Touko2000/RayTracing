#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/23 20:51
# @Author  : lwk
# @Email   : 1293532247@qq.com
# @File    : MyRayTracing.py
# @Software: PyCharm

import numpy as np
from PIL import Image
import time
import datetime

w = 400
h = 300


def normalize(x):
    x /= np.linalg.norm(x)
    return x


# 平面底座
def intersect_plane(O, D, P, N):
    """

    :param O: 点， [float,float,float]
    :param D: 向量
    :param P: 点， [float,float,float]
    :param N: normal 向量，形式为 [float,float,float]
    :return:
    """
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


# 球
def intersect_sphere(O, D, S, R):
    """

    :param O:
    :param D:
    :param S: Position
    :param R: Radius
    :return:
    """
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    # 方程组有解
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


# box
# 对于正方体而言，与光线的交点即为六个面与光线的交点
def intersect_cube(O, D, S, L):
    """

    :param O: 点
    :param D: 向量
    :param S: Position, 该面离原点最近的点
    :param L: Length, 边长
    :return:
    """
    _size = L
    _slabs = []
    normals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    plane_origin = []
    # 构造三个面对, 计算各 Slab
    for _i in range(0, 3):
        planes = [add_plane(S, normals[_i])]
        pointOnFartherPlane = S + [0, 0, 0]
        pointOnFartherPlane[_i] += _size
        plane_origin.append([S, pointOnFartherPlane[_i]])
        planes.append(add_plane(pointOnFartherPlane, normals[_i]))
        _slabs.append(planes)

    # 用基于Slab的正方体求交算法计算长方体与光线的交点
    # 分别计算 3 个 slab与光线的交点
    tMin, tMax = [0., 0., 0.], [0., 0., 0.]
    _normals = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    for _i in range(0, 3):
        t0 = intersect_plane(O, D, np.array(plane_origin[_i][0]), np.array(_slabs[_i][0]['normal']))
        t1 = intersect_plane(O, D, np.array(plane_origin[_i][1]), np.array(_slabs[_i][1]['normal']))
        tMin[_i] = min(t0, t1)
        tMax[_i] = max(t0, t1)

    # 找到最远点和最近点
    tMinMax = tMin[0]
    for _i in range(1, 3):
        if tMin[_i] > tMinMax:
            tMinMax = tMin[_i]
    # tMinMax = min(tMin)
    # tMaxMin = max(tMax)
    tMaxMin = tMax[0]
    for _i in range(0, 3):
        if tMax[_i] < tMaxMin:
            tMaxMin = tMax[_i]

    # 如果最近点比最远点近，则长方体与光线有交点
    if tMinMax < tMaxMin:
        return tMinMax
        # return tMaxMin
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        # 底面
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        # 球体
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'cube':
        # 正方体
        return intersect_cube(O, D, obj['position'], obj['length'])


# 标准化
def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'cube':
        N = normalize(M - obj['position'])
    return N


# 添加颜色
def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


# 光线追踪
def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    l = [intersect(M + N * .0001, toL, obj_sh)
         for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray


# 添加球体
def add_sphere(position, radius, color):
    """

    :param position: 球心
    :param radius: 半径
    :param color: 颜色
    :return:
    """
    return dict(type='sphere', position=np.array(position), radius=np.array(radius), color=np.array(color),
                reflection=.5)


# 添加底面
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (color_plane0 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25)


# 添加长方体
def add_cube(position, length, color):
    """

    :param position: 原点（最靠近坐标原点的顶点）
    :param length: 尺寸（分别与x、y、z轴平行的边的边长）
    :param color: 颜色
    :return:
    """
    return dict(type='cube', position=np.array(position), length=np.array(length), color=np.array(color),
                reflection=.5)


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [1., .572, .184]),
         # add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_cube([-2.75, .1, 3.5], .6, [.5, .223, .5]),
         # add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
         # add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         ]

# 灯光位置和颜色。
L = np.array([6., 6., -10.])
color_light = np.ones(3)

# 默认的灯光和材质参数。
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 10  # 最大的光反射次数。
col = np.zeros(3)  # 当前颜色
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.
img = np.zeros((h, w, 3))

r = float(w) / h
# 屏幕坐标: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# 遍历所有像素
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        # print(type(rayO), type(rayD))
        reflection = 1.
        # 循环穿过初始和次要射线。
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # 反射：创造新的光线。
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

c_time = datetime.datetime.now()
fileend = str(c_time.year) + str(c_time.month) + str(c_time.day) + str(c_time.hour) + str(c_time.minute) + str(
    c_time.second)
picname = "RayTrace" + fileend + ".png"
im = Image.fromarray((255 * img).astype(np.uint8), "RGB")
im.save(picname)
