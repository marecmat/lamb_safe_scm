#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# FEM_1D_porous.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2017
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@kth.se>
# 	Peter Göransson <pege@kth.se>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, det

def dof_us_x(i_e, order):
    ''' Define the usx dof for each element'''
    return slice(i_e*order, 1+(i_e+1)*order)


def dof_us_y(i_e, order, nb_elem):
    ''' Define the usy dof for each element'''
    nb_dof_usx = nb_elem*(order)+1
    return slice(nb_dof_usx+i_e*order, nb_dof_usx+1+(i_e+1)*order)


def build_mat_elem(h):

    m_elem_quad = (h/30)*np.array([[4,2,-1],[2,16,2],[-1,2,4]])
    k_elem_quad = (1/(3*h))*np.array([[7,-8,1],[-8,16,-8],[1,-8,7]])
    c_elem_quad = (1/6)*np.array([[-3,4,-1],[-4,0,4],[1,-4,3]])

    return m_elem_quad, k_elem_quad, c_elem_quad, c_elem_quad.transpose()


def build_global_matrices(nb_elem, L, params, shape_fun='lobatto'):

    lamda, mu, rho = params
    # print(rho, lamda, mu)
    h = L/nb_elem
    n_R = 2*(nb_elem*2 + 1)

    M_elem, K_elem, C1_elem, C2_elem = build_mat_elem(h)

    A_2 = np.zeros((n_R, n_R), dtype=complex)
    A_1 = np.zeros((n_R, n_R), dtype=complex)
    A_0 = np.zeros((n_R, n_R), dtype=complex)
    M   = np.zeros((n_R, n_R), dtype=complex)

    for i_e in range(nb_elem):
        index_u1 = dof_us_x(i_e, 2)
        index_u2 = dof_us_y(i_e, 2, nb_elem)

        A_2[index_u1, index_u1] += (lamda+2*mu)*M_elem
        A_2[index_u2, index_u2] += mu*M_elem
        
        A_1[index_u1, index_u2] += lamda*C1_elem
        A_1[index_u2, index_u1] += mu*C1_elem
        A_1[index_u1, index_u2] -= mu*C2_elem
        A_1[index_u2, index_u1] -= lamda*C2_elem
        
        A_0[index_u2, index_u2] += (lamda+2*mu)*K_elem
        A_0[index_u1, index_u1] += mu*K_elem
        
        M[index_u1, index_u1] += rho*M_elem
        M[index_u2, index_u2] += rho*M_elem

    return A_2/mu, 1j*A_1/mu, A_0/mu, M/mu, n_R


def eigenproblem(freq, nb_elem, L, params):
    om = 2*np.pi*freq

    A2, A1, A0, M, n_R = build_global_matrices(nb_elem, L, params)
        
    I = np.eye(n_R) 
    Z = np.zeros((n_R, n_R))

    A = np.block([[-A1, -A0+(om**2)*M], [I, Z]])
    B = np.block([[A2, Z], [Z, I]])

    k = eigvals(A, B)
    return k


def determinant(freq, k, nb_elem, L, params):
    
    om = 2*np.pi*freq
    A2, A1, A0, M, n_R = build_global_matrices(nb_elem, L, params, shape_fun='lagrange')

    d = np.zeros_like(k)

    for i, kk in enumerate(k):
        K = kk
        full_pb = (K**2*A2 + K*A1 + A0 - om**2*M)
        d[i]= np.abs(det(full_pb))
    
    return d