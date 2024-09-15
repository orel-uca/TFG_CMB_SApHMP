# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:15:57 2024

@author: cemab
"""


def optimiza(n, p, tiempo, a, FC, label):
    # =======================================================
    #       SApHMP-h
    #       Formulación de Alkaabneh et al.
    # =======================================================
    
    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import os
    from sklearn.manifold import MDS
    import math
    
    # Crear el modelo
    modelo = 'SAFLOWLOC'
    mo = Model(modelo)
    
    Q = len(a) # Num. tramos
    L = range(Q)
    
    print()
    print('=======================================================')
    print(modelo)
    print('n= ', n)
    print('p= ', p)
    print('L= ', Q)
    
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
            dist[i,j] = 0.001 * math.sqrt( (P[i][0] - P[j][0])* (P[i][0] - P[j][0]) + (P[i][1] - P[j][1])*(P[i][1] - P[j][1]) );

    O = np.zeros((n))
    for i in N:
        O[i]=W[i,:].sum()
    
    D = np.zeros((n))
    for i in N:
        D[i]=W[:,i].sum()

    # Definir las variables de decisión
    x = mo.addVars([(i, j, k, m) for i in N for j in N for k in N for m in N], vtype=GRB.BINARY, name="x")
    z = mo.addVars([(i, k) for i in N for k in N], vtype=GRB.BINARY, name="z")
    y = mo.addVars([(q, k, m) for q in L for k in N for m in N] , vtype=GRB.BINARY, name="r")
    r = mo.addVars([(q, k, m) for q in L for k in N for m in N] , lb=0.0, name="r")

    # Función objetivo
    mo.setObjective(
    (quicksum(W[i, j]*(dist[i, k]+dist[m, j])*x[i, j, k, m] for i in N for j in N for k in N for m in N)+
        quicksum(dist[k, m]*(a[q]*r[q, k, m]+FC[q]*y[q, k, m]) for q in L for k in N for m in N if k != m)),
        GRB.MINIMIZE
    )
    
    # ---------------------------------------------------------
    #                  RESTRICCIONES
    # ---------------------------------------------------------
    mo.addConstrs(((quicksum(z[i, k] for k in N) == 1) for i in N), name='R1') 
    
    mo.addConstrs((z[i, k] <= z[k, k] for i in N for k in N), name='R3')
    
    mo.addConstr(((quicksum(z[k, k] for k in N) == p)), name='R2')
    
    mo.addConstrs(
        (quicksum(x[i, j, k, m] for m in N) == z[i, k] for i in N for j in N for k in N), name="r5_k"
    )
    mo.addConstrs(
        (quicksum(x[i, j, k, m] for k in N) == z[j, m] for i in N for j in N for m in N), name="r5_m"
    )
    
    mo.addConstrs(
        (quicksum(r[q, k, m] for q in L) == quicksum(W[i, j]*x[i, j, k, m] for i in N for j in N) for k in N for m in N if k != m), name="R13"
    )

    mo.addConstrs(
        ((r[q, k, m] - y[q, k, m]*quicksum(W[i, j] for i in N for j in N)) <= 0 for q in L for k in N for m in N if k != m), name="R13"
    )

    mo.addConstrs(
        (quicksum(y[q, k, m] for q in L) == x[k, m, k, m] for k in N for m in N if k != m), name="R13"
    )
    
    # Parámetros de Gurobi
    mo.Params.TimeLimit = tiempo
    mo.Params.MIPGap = 1e-6
    mo.Params.OutputFlag= 0
    
    mo.update()

    mo.optimize()
    btime = mo.Runtime
    
    # ***************************************************
    #                  salidas
    # ***************************************************
    if not os.path.exists('Salidas'):
        os.makedirs('Salidas')
    
    fichero = 'Salidas/resultados_alka.txt'
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
    
    with open(f'Salidas/val_vars_SAFLOW_{label}_n{n}_p{p}_q{Q}.txt', 'w') as varfile:
        varfile.write("Valores de las variables:\n")
        
        varfile.write("\nz variables:\n")
        for i in N:
            for k in N:
                if z[i, k].x != 0:
                    varfile.write(f"z[{i}, {k}] = {z[i, k].x}\n")
                        
        varfile.write("\nx variables:\n")                
        for i in N:
            for j in N:
                for k in N:
                    for m in N:
                        if x[i, j, k, m].x != 0:
                            varfile.write(f"x[{i}, {j}, {k}, {m}] = {x[i, j, k, m].x}\n")
                            
        varfile.write("\ny variables:\n")
        
        for q in L:
            for k in N:
                for m in N:
                    if k!=m:
                        if y[q, k, m].x != 0:
                            varfile.write(f"y[{q}, {k}, {m}] = {y[q, k, m].x}\n")

        varfile.write("\nr variables:\n")
        for q in L:
            for k in N:
                for m in N:
                    if k!=m and r[q, k, m].x != 0: 
                        varfile.write(f"r[{q}, {k}, {m}] = {r[q, k, m].x}\n")

    varfile.close()
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coordinates = mds.fit_transform(dist)
    
    # Guardar las coordenadas y nodos hubs
    with open(f'Salidas/coordenadas_SAFLOW_{label}_n{n}_p{p}.txt', 'w') as coordfile:
        coordfile.write("Coordenadas de los nodos:\n")
        for i, (x_coord, y_coord) in enumerate(coordinates):
            coordfile.write(f"Nodo {i+1}: ({x_coord}, {y_coord})\n")
        
        coordfile.write("\nNodos seleccionados como hubs:\n")
        for k in N:
            if z[k,k].x > 0.5:  # Si el nodo es seleccionado como hub
                coordfile.write(f"Hub {k+1}\n")
    coordfile.close()

