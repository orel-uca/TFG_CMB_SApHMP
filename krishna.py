#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:15:59 2021

@author: raulpaez
"""


def optimiza(n, p, tiempo):

    # =======================================================
    #       P-HUBS
    #       Formulación de Krishnamoorthy
    # =======================================================

    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import os
    from sklearn.manifold import MDS
    import math 
    
    modelo = 'P-HUBS-K'
    print()
    print('=======================================================')
    print(modelo)
    print('n= ', n)
    print('p= ', p)


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

    # ------------------------------------------------------------------------
    #              CREATE A NEW MODEL AND VARIABLES
    # ------------------------------------------------------------------------
    mo = Model(modelo)

    # ------------------------  variables de matching  -----------------------
    # z = mo.addVars([(k) for k in N], vtype=GRB.BINARY, name="z")
    x = mo.addVars([(i, k)  for i in N for k in N ], vtype=GRB.BINARY, name="x")
    y = mo.addVars([(k, m, i)  for k in N for m in N for i in N], lb=0.0, name="y")

    # -----------------definiendo objetivo------------------------------------

    mo.setObjective((quicksum(0.75*dist[k, m]*y[ k, m, i]  for k in N for m in N for i in N) +
                    quicksum(dist[i, k]*(O[ i]+D[ i])*x[ i, k]  for i in N for k in N)), 
                    GRB.MINIMIZE)

    # ---------------------------------------------------------
    #                  CONSTRAINTS
    # ----------------------------------------------------------
    mo.addConstrs(((quicksum(x[ i, k] for k in N) == 1) for i in N), name='R1')
    mo.addConstr(((quicksum(x[k,k] for k in N) == p)), name='R2')
    mo.addConstrs((x[i, k] <= x[k,k]  for i in N for k in N ), name='R3')
    mo.addConstrs(((quicksum(y[k, m, i] for m in N) - quicksum(y[ m, k, i] for m in N)) == (O[i]*x[i, k] - quicksum(W[i, j]*x[ j, k] for j in N)) for i in N for k in N), name='R5')
    
    
    # Para caso capacidades
    # mo.addConstrs((quicksum(y[s, k, m, i] for m in N) <= O[s, i]*x[s, i, k] for s in S for i in N for k in N if i!=k), name='R6')
    
    # ***********************************************
    #               Resolviendo
    # ***********************************************

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

    fichero = 'Salidas/resultados_kris.txt'
    outfile = open(fichero, 'a')

    # 3=Infactible, 4=infactible o no acotado y 5=no acotado
    if mo.status == 3:
        print('Modelo no resuelto por ser infactible')
        print()
        print('{:17} {:5} {:5}   Infactible'.format(modelo, n, p), file=outfile)
        return

    if mo.status == 4:
        print('Modelo no resuelto por ser infactible o no acotado')
        print()
        print('{:17} {:5} {:5}   Infact.-No_acotado'.format(modelo, n, p), file=outfile)
        return

    if mo.status == 5:
        print('Modelo no resuelto por ser no acotado')
        print()
        print('{:17} {:5} {:5}   No_acotado'.format(modelo, n, p), file=outfile)
        return

    if mo.status == GRB.Status.OPTIMAL:
        print("TIME: %f" % btime)
        print("OV: %g " % mo.ObjVal)
        # for m in N:
        #     for k in N:
        #         for s in S:
        #             if y[s,k,m,1].x > 0.0:
        #                 print("y[%d,%d,%d,%d]=%f;" % (s,k,m,i,y[s,k,m,1].x))

        print('{:12} {:4} {:3}        {:8.1f}      {:5.1f}   {:5.2f}  {:7d}'.format(modelo, n, p, mo.ObjVal, btime, mo.MIPGap, int(mo.NodeCount)), file=outfile)
    else:
        print('{:12} {:4} {:3}        {:8.1f}      {:5.1f}   {:5.2f}  {:7d}'.format(modelo, n, p, mo.ObjVal, btime, mo.MIPGap, int(mo.NodeCount)), file=outfile)
    
    outfile.close()
    
    with open(f'Salidas/val_vars_kris_n{n}_p{p}.txt', 'w') as varfile:
        varfile.write("Valores de las variables:\n")

        varfile.write("\nx variables:\n")
        for i in N:
            for k in N:
                if x[i,k].x != 0:
                    varfile.write(f"x[ {i}, {k}] = {x[i, k].x}\n")
        
        varfile.write("\ny variables:\n")
        for k in N:
            for m in N:
                for i in N:
                    if y[k,m,i].x != 0:
                        varfile.write(f"y[ {k}, {m}, {i}] = {y[ k, m, i].x}\n")
    
    varfile.close()
    
        # Generar las coordenadas con MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coordinates = mds.fit_transform(dist)
    
    # Guardar las coordenadas y nodos hubs
    with open(f'Salidas/coordenadas_Kris_n{n}_p{p}.txt', 'w') as coordfile:
        coordfile.write("Coordenadas de los nodos:\n")
        for i, (x_coord, y_coord) in enumerate(coordinates):
            coordfile.write(f"Nodo {i+1}: ({x_coord}, {y_coord})\n")
        
        coordfile.write("\nNodos seleccionados como hubs:\n")
        for k in N:
            if x[k, k].x > 0.5:  # Si el nodo es seleccionado como hub
                coordfile.write(f"Hub {k+1}\n")
                
    coordfile.close()
