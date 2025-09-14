# Designing an optimized fueling infrastructure for a hydrogen-powered railway system
# Author: Alessio Trivella

from math import radians, sin, cos, sqrt, atan2, floor, ceil
from shapely.geometry import Polygon, Point
import time
import gurobipy as gp
from gurobipy import GRB
import read_and_print as rap
import models_and_algo as algo

# computes distance based on latitude and longitude coordinates
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0 # Radius of the Earth in meters
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Calculate the differences between latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Calculate the distance
    distance = R * c
    return distance

# produces a grid of points within a shape (polygon) 
def discretize_area(polygon_coords, STEP_TT, PLOT_AREA, yard, area):
    polygon = Polygon(polygon_coords) # construct polygon from coordinates

    min_lati, min_longi, max_lati, max_longi = polygon.bounds # bounding box coordinates
    dist_lati = haversine_distance(min_lati,min_longi,max_lati,min_longi) # latitude distance [m]
    dist_longi = haversine_distance(min_lati,min_longi,min_lati,max_longi) # longitude distance [m]
    delta_lati = max_lati - min_lati # latitude distance [degrees]
    delta_longi = max_longi - min_longi # longitude distance [degrees]
    N_lati = ceil(dist_lati / STEP_TT) + 1 # discretization of x
    N_longi = ceil(dist_longi / STEP_TT) + 1 # discretization of y
    STEP_deg_lati = delta_lati / (N_lati-1) # spacing in latitude degrees
    STEP_deg_longi = delta_longi / (N_longi-1) # spacing in longitude degrees

    points = [] # initialize empty list to store generated points
    for p in range(0,len(polygon_coords)): # add corners to point set
        points.append((polygon_coords[p][0], polygon_coords[p][1]))

    # generate points in a regular grid within the bounding box
    for i in range(0,N_lati): 
        for j in range(0,N_longi):
            x = min_lati + i * STEP_deg_lati
            y = min_longi + j * STEP_deg_longi
            new_point = (x, y)
            if polygon.contains(Point(new_point)): # check if inside polygon
                points.append(new_point) # add to list
    if PLOT_AREA == 1:
        rap.plot_discretized_area(polygon,points,dist_lati,dist_longi,yard,area)

    return points

# discretizes a line
def discretize_line(L_lati, L_longi, STEP_FFI, PLOT_LINE, yard, line):
    
        lati_1 = L_lati[0] # extremes of the line
        lati_2 = L_lati[1]   
        longi_1 = L_longi[0] 
        longi_2 = L_longi[1] 

        length_line = haversine_distance(lati_1, longi_1, lati_2, longi_2)
        N_disc = max(2, floor(length_line / STEP_FFI) + 1) # number of points including extremes
        delta_lati = (lati_2 - lati_1) / (N_disc - 1)
        delta_longi = (longi_2 - longi_1) / (N_disc - 1)

        points = [] 
        for n in range(0,N_disc): 
            x = lati_1 + n*delta_lati
            y = longi_1 + n*delta_longi
            points.append((x,y)) # add point to discretized point list

        if PLOT_LINE == 1:
            rap.plot_discretized_line(L_lati, L_longi, points, yard, line)

        return points

# discretize areas, lines, and return lists of all locations - based on DISTANCE AMONG POINTS
def determine_locations(Ir,Kr,items,A_lati,A_longi,L_lati,L_longi,
      P_lati,P_longi,STEP_TT,STEP_FFI,PLOT_AREA,PLOT_LINE,C,F,G):
    start_time = time.time()

    print("\nCompute TT locations")
    J = [] # number of TT locations
    lati_ij = [] # TT location coordinates
    longi_ij = [] # TT location coordinates
    for i in Ir: # yards
        lati_i = [] # TT latitude in yard i
        longi_i = [] # TT longitude in yard i
        for a in range (0, items[i][0]): # areas of yard i
            polygon_coords = list(zip(A_lati[i][a],A_longi[i][a])) # polygon perimeter
            points = discretize_area(polygon_coords, STEP_TT, PLOT_AREA, i, a)
            for p in range(0,len(points)): 
                lati_i.append(points[p][0]) # add point to list
                longi_i.append(points[p][1])
        lati_ij.append(lati_i)
        longi_ij.append(longi_i)
        J.append(len(lati_ij[i]))
        print("Yard %d: %d points for TT" %(i,J[i]))

    H = [] # number of FFD locations
    lati_ikh = [] # FD location coordinates
    longi_ikh = [] # FD location coordinates

    print("Compute FD locations")
    for i in Ir: # yards
        lati_i = []
        longi_i = []
        H_i=[]
        for k in Kr:
            lati_ik=[]
            longi_ik=[]
            if k == 0: # FFD
                for l in range (0, items[i][1]): 
                    points = discretize_line(L_lati[i][l], L_longi[i][l], STEP_FFI, PLOT_LINE, i, l)
                    for p in range(0,len(points)): 
                        lati_ik.append(points[p][0])
                        longi_ik.append(points[p][1])
            if k == 1: # SFD
                for p in range (0, items[i][2]):
                    lati_ik.append(P_lati[i][p])
                    longi_ik.append(P_longi[i][p])
            lati_i.append(lati_ik)
            longi_i.append(longi_ik)
            H_i.append(len(lati_ik))
        lati_ikh.append(lati_i)
        longi_ikh.append(longi_i)
        H.append(H_i)   

    for i in Ir: # yards
        for k in Kr:    
            print("Yard %d: %d points for FD %d" %(i,H[i][k],k))

    print(" --> Time to compute coordinates: %.2fs" %(time.time() - start_time)) 

    Mi = [] # big-M used for yard capacity
    for i in Ir:
        Mi.append(2*C[i])
    Mk = [] # big-M used for pipeline-flow
    for k in Kr:
        Mk.append(max(F,G[k]))

    # J and H ranges
    Jr = []
    Hr = []
    for i in Ir:
        Jr.append(range(0,J[i]))
        Hr_row = []
        for k in Kr:
            Hr_row.append(range(0,H[i][k]))
        Hr.append(Hr_row)

    return J,Jr,lati_ij,longi_ij,H,Hr,lati_ikh,longi_ikh,Mi,Mk

# discretize areas, lines, and return lists of all locations - based on NUMBER OF POINTS
def determine_locations_balanced(Ir,Kr,items,A_lati,A_longi,L_lati,L_longi,
      P_lati,P_longi,POINTS,PLOT_AREA,PLOT_LINE,C,F,G):
    start_time = time.time()
    STEP = [50] * len(Ir) # discretization step per yard
    LOC = [0] * len(Ir) # total number of locations per yard
    FLAG = [0] * len(Ir) # when flag is active, stop search for yard
    DIVIDED = [0] * len(Ir) # number of times step is divided
    TOL = 5 # 5 step divisions

    print("\nCompute balanced locations")
    J = [] # number of TT locations
    lati_ij = [] # TT location coordinates
    longi_ij = [] # TT location coordinates
    H = [] # number of FFD locations
    lati_ikh = [] # FD location coordinates
    longi_ikh = [] # FD location coordinates

    for i in Ir: # yards
        print("\nDiscretizing YARD %d..." %i)
        while FLAG[i] == 0:

            # TT
            lati_i1 = [] # TT latitude in yard i
            longi_i1 = [] # TT longitude in yard i
            for a in range (0, items[i][0]): # areas of yard i
                polygon_coords = list(zip(A_lati[i][a],A_longi[i][a])) # polygon perimeter
                points = discretize_area(polygon_coords, STEP[i], PLOT_AREA, i, a)
                for p in range(0,len(points)): 
                    lati_i1.append(points[p][0]) # add point to list
                    longi_i1.append(points[p][1])
            LOC[i] = len(lati_i1) 

            # FD
            lati_i2 = []
            longi_i2 = []
            H_i=[]
            for k in Kr:
                lati_ik=[]
                longi_ik=[]
                if k == 0: # FFD
                    for l in range (0, items[i][1]): 
                        points = discretize_line(L_lati[i][l], L_longi[i][l], STEP[i], PLOT_LINE, i, l)
                        for p in range(0,len(points)): 
                            lati_ik.append(points[p][0])
                            longi_ik.append(points[p][1])
                if k == 1: # SFD
                    for p in range (0, items[i][2]):
                        lati_ik.append(P_lati[i][p])
                        longi_ik.append(P_longi[i][p])
                lati_i2.append(lati_ik)
                longi_i2.append(longi_ik)
                H_i.append(len(lati_ik))
                LOC[i] = LOC[i] + H_i[k]

            # print(LOC, end = ""), print(STEP)
            if LOC[i] >= POINTS and DIVIDED[i] >= TOL:
                FLAG[i] = 1
                lati_ij.append(lati_i1)
                longi_ij.append(longi_i1)
                J.append(len(lati_ij[i]))
                lati_ikh.append(lati_i2)
                longi_ikh.append(longi_i2)
                H.append(H_i)
            elif LOC[i] >= POINTS:
                STEP[i] = STEP[i] + 1/ (2 ** DIVIDED[i]) 
                DIVIDED[i] = DIVIDED[i] + 1
            else:
                STEP[i] = STEP[i] - 1/ (2 ** DIVIDED[i])

        print("Results yard %d: %d points TT" %(i,J[i])) 
        for k in Kr: print("Yard %d: %d points FD %d" %(i,H[i][k],k))

    print("Final number of locations and step: ", end = ""), print(LOC, end = ""), print(STEP)
    print(" --> Time to compute balanced coordinates: %.2fs" %(time.time() - start_time)) 

    Mi = [] # big-M used for yard capacity
    for i in Ir:
        Mi.append(2*C[i])
    Mk = [] # big-M used for pipeline-flow
    for k in Kr:
        Mk.append(max(F,G[k]))

    # J and H ranges
    Jr = []
    Hr = []
    for i in Ir:
        Jr.append(range(0,J[i]))
        Hr_row = []
        for k in Kr:
            Hr_row.append(range(0,H[i][k]))
        Hr.append(Hr_row)

    return J,Jr,lati_ij,longi_ij,H,Hr,lati_ikh,longi_ikh,Mi,Mk

# compute distances between each facility pair in a yard
def compute_distances(lati_ij,longi_ij,lati_ikh,longi_ikh,Ir,Jr,Kr,Hr,F,G,beta,gamma):
    start_time = time.time()
    L = [] # list-based 3D matrix L[i][l][feature] with all locations in a yard
    E = [] # list-based 3D matrix E[i][l][m] with distances between L locations
    L_split = [] # number of TT locations
    Lr = [] # list of ranges (number of locations)
    LTTr = [] # list of ranges for TT locations
    LFDr =[]  # list of ranges for FD locations

    print("\nCompute distance matrix")
    for i in Ir: # select yard
        print("Yard %i..." %i)
        L_i = []
        for j in Jr[i]: # select TT location (last element: type = 0)
            L_i.append([lati_ij[i][j], longi_ij[i][j], 0, F, beta[i], 0])
        L_split.append(len(L_i))
        for k in Kr: # select FD type
            for h in Hr[i][k]: # select FD location (last element: FFD type = 1; SFD type = 2)
                L_i.append([lati_ikh[i][k][h], longi_ikh[i][k][h], 1, G[k], gamma[i][k], k+1])
        L.append(L_i)
        LTTr.append(range(0,L_split[i]))
        LFDr.append(range(L_split[i]+1,len(L_i)))
        Lr.append(range(0,len(L_i)))

        # distance matrix
        E_i = []
        for l in range(0,len(L_i)):
            E_il = []
            for m in range(0,len(L_i)):
                E_il.append(haversine_distance(L_i[l][0], L_i[l][1], L_i[m][0], L_i[m][1]))
            E_i.append(E_il)
        E.append(E_i)

    print(" --> Time to compute distances: %.2fs\n" %(time.time() - start_time)) 
    return E, L, Lr, LTTr, LFDr

# test different configuration heuristic settings (phase 1) to determine the best (cost, runtime)
def tuning_configurations_setting (LTTr,LFDr,E,L,N,C,Ir,D_MAT):
    MAX_FAC_VAL = [2,3] # maximum number of facilities in a configuration ([2,3,4,5,6,7,8])
    SCALE_VAL = [0.4,0.5,0.8] # percentage of number of pairs ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] )
    total_sum_ALL = []
    runtime_ALL = []
    for s1 in range(len(MAX_FAC_VAL)):
        total_sum_ALL_i = []
        runtime_ALL_i = []
        for s2 in range(len(SCALE_VAL)):
            MAX_FAC = MAX_FAC_VAL[s1]
            SCALE = SCALE_VAL[s2]
            start_time = time.time()
            print("-----------------------------------RUN PHASE 1 -------------------------------------")
            cost, config, edges, chosen = [], [], [], [] # per-yard cost and configurations from heuristic
            for i in Ir:
                cost_i, config_i, edges_i, chosen_i, FLAG = algo.configuration_heuristic_v3(LTTr[i],LFDr[i],E[i],L[i],min(N,C[i]),SCALE,D_MAT,MAX_FAC)
                cost.append(cost_i), config.append(config_i), edges.append(edges_i), chosen.append(chosen_i)
                if FLAG != 2: # check if all optimization problems in phase 1 were feasible
                    print("Phase 1: (MAX_FAC,SCALE) = (", MAX_FAC,", ",SCALE,") --> INFEASIBLE (flag ",FLAG,")\n")
                    total_sum_ALL_i.append(-1)
                    runtime_ALL_i.append(-1)
                    break
            if FLAG == 2:
                total_sum = sum(value for row in cost for value in row)
                runtime = time.time() - start_time
                print("Phase 1: (MAX_FAC, SCALE) = (", MAX_FAC,",",SCALE,")")
                print("Total cost: ", total_sum)
                print("Runtime: ", runtime,"\n")
                total_sum_ALL_i.append(total_sum)
                runtime_ALL_i.append(runtime)
                #print_2D_list(cost,"Costs from phase 1\n") # show all costs for subproblems
        total_sum_ALL.append(total_sum_ALL_i)
        runtime_ALL.append(runtime_ALL_i)
    print("============================== FULL RESULTS TUNING ==============================")
    rap.print_2D_list(total_sum_ALL,"Costs all experiments\n") # show all costs for subproblems
    rap.print_2D_list(runtime_ALL,"Runtime all experiments\n") # show all costs for subproblems

# extract the heuristic solution from phases 1 and 2
def extract_solution_Ph1_Ph2(Lr,x,v_sum,config,edges,chosen,Ir):
    y_H, z_H, u_H, w_H = [], [], [], [] # extract the heuristic solution 
    for i in Ir:
        y_Hi, z_Hi = [0]*len(Lr[i]), [0]*len(Lr[i]) # facilities
        u_Hi, w_Hi = [[0] * len(Lr[i]) for _ in range(len(Lr[i]))], [[0] * len(Lr[i]) for _ in range(len(Lr[i]))] # pipelines
        if x[i] > 0.5:
            for j in range(0, len(chosen[i][int(v_sum[i])])):
                c = chosen[i][int(v_sum[i])][j] # configuration
                for l in range(0,config[i][c][5]): # facilities in configuration
                    y_Hi[config[i][c][6+l]] = 1 # select facility
                for e in range(0, len(edges[i][c])): # edges in configuration
                    u_Hi[edges[i][c][e][0]][edges[i][c][e][1]] = 1 # select connection
        y_H.append(y_Hi) # add yard vector to overall solution
        z_H.append(z_Hi) # empty vector
        u_H.append(u_Hi) # add yard matrix to overall solution
        w_H.append(w_Hi) # empty matrix
    return y_H, z_H, u_H, w_H