# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:14:12 2024

@author: cemab
"""


def optimiza(n, p, tiempo, a, FC, label):
    # =======================================================
    #       SApHMP-h
    #       Formulación de Rostami et al.
    # =======================================================
    
    
    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import os
    from sklearn.manifold import MDS
    import math
    # import nolineal as nl

    Q = len(a) # Num tramos
    L = range(Q)
    
    modelo = 'Rostami'
    print()
    print('=======================================================')
    print(modelo)
    print('n= ', n)
    print('p= ', p)
    print('L= ', Q)
    
    # Definir los nodos
    dataP = np.zeros((n,2))
    dataW = np.zeros((n,n))
    
    # Cargamos los datos
    aux1 = np.loadtxt('./Datos_phub/' + str(n) + '_' + '4' + '.txt', skiprows=1, max_rows=n)
    dataP[:, :] = aux1
    
    aux2 = np.loadtxt('./Datos_phub/' + str(n) + '_' + '4' + '.txt', skiprows=n+2, max_rows=n)
    dataW[:, :] = aux2
    
    N = range(n)
    
    P = np.zeros((n, 2))
    for i in N:
        for j in range(2):
            P[i, j] = dataP[i,j]
    
    W = np.zeros((n, n))
    for i in N:
        for j in N:
            W[i, j] = dataW[i,j]
            
    dist = np.zeros((n, n))
    for i in N:
        for j in N:
            dist[i,j] = 0.001 * math.sqrt( (P[i][0] - P[j][0])* (P[i][0] - P[j][0]) + (P[i][1] - P[j][1])*(P[i][1] - P[j][1]) )
    
    O = np.zeros((n))
    for i in N:
        O[i]=W[i,:].sum()
    
    D = np.zeros((n))
    for i in N:
        D[i]=W[:,i].sum()

    # ------------------------------------------------------------------------
    #              CREATE A NEW MODEL AND VARIABLES
    # ------------------------------------------------------------------------
    mo = Model(modelo)
    
    # ------------------------   Variables de decisión  -----------------------
    x = mo.addVars([(i, k) for i in N for k in N], vtype=GRB.BINARY, name="x")
    y = mo.addVars([(q, i, k, l) for q in L for i in N for k in N for l in N], lb=0.0, name="y")
    z = mo.addVars([(q, k, l) for q in L for k in N for l in N], vtype=GRB.BINARY, name="z")
    
    # -----------------definiendo objetivo------------------------------------
    mo.setObjective(
        (quicksum(dist[i, k]*(O[i]+D[i])*x[i, k] for i in N for k in N ) +
        quicksum(dist[k, l]*FC[q] * z[q, k, l] for q in L for k in N for l in N)+
        quicksum(dist[k, l]*a[q] * y[q, i, k, l] for q in L for k in N for l in N for i in N )),
        GRB.MINIMIZE
    )

    # ---------------------------------------------------------
    #                  CONSTRAINTS
    # ----------------------------------------------------------
    mo.addConstrs(((quicksum(x[i, k] for k in N) == 1) for i in N), name='R1')
    mo.addConstr(((quicksum(x[k, k] for k in N) == p)), name='R2')
    mo.addConstrs((x[i, k] <= x[k, k] for i in N for k in N ), name='R3')
    mo.addConstrs((quicksum(z[q, k, l] for q in L) >= x[k, k] + x[l, l] - 1 for k in N for l in N if k != l), name='rost1')

    mo.addConstrs((quicksum(y[q, i, k, l] for q in L for l in N) == O[i]*x[i, k] for i in N for k in N ), name='R6')
    mo.addConstrs((quicksum(y[q, i, l, k] for q in L for l in N) == quicksum(W[i, j] * x[j, k] for j in N) for i in N for k in N), name='rost2')
    
    #mo.addConstrs((quicksum(y[q, i, k, l] for i in N) <= U[q] * z[q, k, l] for q in L for k in N for l in N if k != l), name='rost3')
    mo.addConstrs((quicksum(y[q, i, k, l] for i in N) <= z[q, k, l] * quicksum(W[i, j] for i in N for j in N) for q in L for k in N for l in N if k !=l), name='rost_new')

    # Parámetros de Gurobi
    mo.Params.TimeLimit = tiempo
    mo.Params.MIPGap = 1e-6
    mo.Params.OutputFlag= 1

    mo.update()
    
    mo.optimize()
    btime = mo.Runtime
    
    # ***************************************************
    #                  salidas
    # ***************************************************
    if not os.path.exists('Salidas'):
        os.makedirs('Salidas')
    
    fichero = 'Salidas/resultados_rost.txt'
    outfile = open(fichero, 'a')
    
    # 3=Infactible, 4=infactible o no acotado y 5=no acotado
    if mo.status == 3:
        print('Modelo no resuelto por ser infactible')
        print()
        print('{:17} {:4} {:3} {:4}   Infactible'.format(modelo, label, n, p), file=outfile)
        return
    
    if mo.status == 4:
        print('Modelo no resuelto por ser infactible o no acotado')
        print()
        print('{:17} {:4} {:3} {:4}   Infact.-No_acotado'.format(modelo, label, n, p), file=outfile)
        return
    
    if mo.status == 5:
        print('Modelo no resuelto por ser no acotado')
        print()
        print('{:17} {:4} {:3} {:4}   No_acotado'.format(modelo, label, n, p), file=outfile)
        return
    
    if mo.status == GRB.Status.OPTIMAL:
        print("TIME: %f" % btime)
        print("OV: %g " % mo.ObjVal)
        print('{:12} {:4} {:3} {:4}        {:8.1f}      {:5.1f}   {:5.2f}  {:7d}'.format(modelo, label, n, p, mo.ObjVal, btime, mo.MIPGap, int(mo.NodeCount)), file=outfile)
    else:
        print('{:12} {:4} {:3} {:4}        {:8.1f}      {:5.1f}   {:5.2f}  {:7d}'.format(modelo, label, n, p, mo.ObjVal, btime, mo.MIPGap, int(mo.NodeCount)), file=outfile)
    
    outfile.close()
    
    # Aquí guardamos los valores de las variables en un fichero separado
    with open(f'Salidas/val_vars_Rost_{label}_n{n}_p{p}_q{Q}.txt', 'w') as varfile:
        varfile.write("Valores de las variables:\n")
        
        varfile.write("\nx variables:\n")
        for i in N:
            for k in N:
                if x[i, k ].x !=0:
                    varfile.write(f"x[{i}, {k}] = {x[i, k].x}\n")
        
        varfile.write("\ny variables:\n")
        for q in L:
            for i in N:
                for k in N:
                    for l in N:
                            if y[q, i, k, l].x != 0:
                                varfile.write(f"y[{q}, {i}, {k}, {l}] = {y[q, i, k, l].x}\n")
        
        varfile.write("\nz variables:\n")
        for q in range(Q):
            for k in N:
                for l in N:
                    if z[q, k, l].x != 0:
                        varfile.write(f"z[{q}, {k}, {l}] = {z[q, k, l].x}\n")
    
    varfile.close()
    
    # Generar las coordenadas con MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coordinates = mds.fit_transform(dist)
    
    # Guardar las coordenadas y nodos hubs
    with open(f'Salidas/coordenadas_Rost_{label}_n{n}_p{p}.txt', 'w') as coordfile:
        coordfile.write("Coordenadas de los nodos:\n")
        for i, (x_coord, y_coord) in enumerate(coordinates):
            coordfile.write(f"Nodo {i+1}: ({x_coord}, {y_coord})\n")
        
        coordfile.write("\nNodos seleccionados como hubs:\n")
        for k in N:
            if x[k, k].x > 0.5:  # Si el nodo es seleccionado como hub
                coordfile.write(f"Hub {k+1}\n")

    coordfile.close()
