# Designing an optimized fueling infrastructure for a hydrogen-powered railway system
# Author: Alessio Trivella

import gurobipy as gp
from gurobipy import GRB
import random
import read_and_print as rap

############## FULL MIP ##############

# solve the full MIP model using Gurobi
def solveMIP(solution_file_path,I,S,Ir,Nr,Lr,LTTr,LFDr,E,L,Dn,alpha,rho,delta,Kr,Sn,C,F,G,D_MAT,MODEL_gap,MODEL_runtime):
    model = gp.Model('model_full')
    EPS = 0.1 # to model min flow in the constraints

    # Decision Variables
    x = model.addVars(I, vtype=GRB.BINARY, name="x")
    v = model.addVars(((i, n) for i in Ir for n in Nr), vtype=GRB.BINARY, name="v")
    # location variables
    y = model.addVars(((i, l) for i in Ir for l in Lr[i]), vtype=GRB.BINARY, name="y")
    z = model.addVars(((i, l) for i in Ir for l in Lr[i]), vtype=GRB.CONTINUOUS, name="z")
    # connection variables
    u = model.addVars(((i, l, m) for i in Ir for l in Lr[i] for m in Lr[i]), vtype=GRB.BINARY, name="u")
    w = model.addVars(((i, l, m) for i in Ir for l in Lr[i] for m in Lr[i]), vtype=GRB.CONTINUOUS, name="w")

    # Objective Function
    obj_expr = (
        gp.quicksum(alpha[i] * x[i] for i in Ir) + # Term [C1]
        2 * rho * gp.quicksum(Dn[i][n] * v[i, n] for i in Ir for n in Nr) + # Term [C2]
        gp.quicksum(L[i][l][4] * y[i, l] for i in Ir for l in Lr[i]) + # Term [C3]
        delta * gp.quicksum(E[i][l][m] * u[i, l, m] for i in Ir for l in Lr[i] for m in Lr[i]) # Term [C4]
    )
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints
    model.addConstrs(gp.quicksum(v[i, n] for n in Nr) <= C[i] * x[i] for i in Ir)  # Yard capacity respected
    model.addConstrs(gp.quicksum(v[i, n] for i in Ir) == 1 for n in Nr)  # Each train is assigned to a yard
    # total supply and demand
    model.addConstrs(gp.quicksum(z[i, l] for l in LTTr[i]) == gp.quicksum(v[i, n] for n in Nr) for i in Ir)  
    model.addConstrs(gp.quicksum(z[i, l] for l in LFDr[i]) == gp.quicksum(v[i, n] for n in Nr) for i in Ir) 
    model.addConstrs( # demand/supply iff facility is chosen
        EPS * y[i, l] <= z[i, l] for i in Ir for l in Lr[i] )  
    model.addConstrs( # demand/supply iff facility is chosen
        z[i, l] <= L[i][l][3] * y[i, l] for i in Ir for l in Lr[i] )  
    model.addConstrs( # flow iff pipeline is built
        EPS * u[i, l, m] <= w[i, l, m] for i in Ir for l in Lr[i] for m in Lr[i])  
    model.addConstrs( # flow iff pipeline is built
        w[i, l, m] <= max(*G, F) * u[i, l, m] for i in Ir for l in Lr[i] for m in Lr[i]) 
        # w[i, l, m] <= min(L[i][l][3],L[i][m][3]) * u[i, l, m] for i in Ir for l in Lr[i] for m in Lr[i])   
    #  flow balance
    model.addConstrs( gp.quicksum(w[i, l, m] for m in Lr[i]) - gp.quicksum(w[i, m, l] for m in Lr[i])
                     == (1 - 2*L[i][l][2]) * z[i, l] for i in Ir for l in Lr[i] )  # demand/supply iff facility is chosen
    model.addConstrs(u[i, l, m] + u[i, m, l] <= 1 for i in Ir for l in Lr[i] for m in Lr[i])  

    for i in Ir:
        for l in Lr[i]:
            u[i, l, l].ub = 0
            # model.addConstr(u[i, l, l] == 0)

    # min distance constraints
    for i in Ir:
        for l in Lr[i]:
            for m in Lr[i]:
                if l != m:
                    if (E[i][l][m] < D_MAT[L[i][l][5]][L[i][m][5]]):    
                        model.addConstr(y[i, l] + y[i, m] <= 1)                

    # Optimize the model
    model.Params.MIPGap = MODEL_gap
    model.Params.TimeLimit = MODEL_runtime
    model.optimize()

    # Print solution on .txt file
    if model.status == GRB.OPTIMAL or GRB.TIME_LIMIT:
        rap.print_solution(solution_file_path[0],x,v,y,z,w,u,Ir,Kr,I,S,Lr,LTTr,LFDr,
                           model.MIPGap,model.Runtime,alpha,delta,rho,Dn,Nr,L,E,Sn,G)
        rap.print_solution_Matlab(solution_file_path[1],x,v,y,z,w,u,Ir,I,L,Lr,LTTr,LFDr,model.objVal,
                                  alpha,delta,rho,Dn,Nr,E)
    else:
        print("No solution found.")
        
    return x,v,y,z,w,u

############## MATHEURISTIC ##############

# solve the first-phase (train alocation) optimization problem
def MIP_phase1(I,Ir,Nr,C,cost,rho,alpha,Dn):
    mPH1 = gp.Model('model_PH1')
    mPH1.setParam('OutputFlag', 0)

    # Decision variables
    x = mPH1.addVars(I, vtype=GRB.BINARY, name="x")
    v = mPH1.addVars(((i, n) for i in Ir for n in Nr), vtype=GRB.BINARY, name="v")
    g = mPH1.addVars(((i, n) for i in Ir for n in range(0,C[i]+1)), vtype=GRB.BINARY, name="g") # support

    # constraints
    mPH1.addConstrs(gp.quicksum(v[i, n] for n in Nr) <= C[i] * x[i] for i in Ir) # Yard capacity respected
    mPH1.addConstrs(gp.quicksum(v[i, n] for i in Ir) == 1 for n in Nr) # Each train is assigned to a yard
    mPH1.addConstrs(x[i] <= gp.quicksum(v[i, n] for n in Nr)  for i in Ir) # v=0 -> x=0

    # Objective function and constraints
    obj_expr = (
        gp.quicksum(alpha[i] * x[i] for i in Ir) + # Term [C1]
        2 * rho * gp.quicksum(Dn[i][n] * v[i, n] for i in Ir for n in Nr) # Term [C2]
        + gp.quicksum(cost[i][n] * g[i, n] for i in Ir for n in range(0,C[i]+1))) # adaptation
    mPH1.addConstrs(gp.quicksum(g[i, n] for n in range(0,C[i]+1)) == 1 for i in Ir) 
    mPH1.addConstrs(gp.quicksum(v[i, m] for m in Nr) == gp.quicksum( n * g[i, n] for n in range(0,C[i]+1)) for i in Ir) 
    
    # Optimize model
    mPH1.setObjective(obj_expr, GRB.MINIMIZE) 
    mPH1.update()
    mPH1.optimize()

    # extract objective and variable values
    obj = []
    obj.append(mPH1.objVal) # full objective
    obj.append( sum(alpha[i] * x[i].x for i in Ir) # allocation cost
        + 2 * rho * sum((Dn[i][n])* v[i, n].x for i in Ir for n in Nr)) 
    x_opt = []
    v_opt = [0] * I
    v_full = []
    for i in Ir:
        x_opt.append(x[i].x) 
        v_full_i = []
        for n in Nr:
            v_opt[i] += v[i, n].x
            v_full_i.append(v[i, n].x)
        v_full.append(v_full_i)

    return obj,x_opt,v_opt,v_full

# solve the second-phase (yard-specific) optimization problem
def MIP_phase2(Lr_i,LTTr_i,LFDr_i,E_i,L_i,Vi,delta,F,G,D_MAT,MODEL_gap,MODEL_runtime,WARM,u_H,y_H):

    mPH2 = gp.Model('model_PH2')
    EPS = 0.1 # to model min flow in the constraints

    # Decision Variables
    y = mPH2.addVars((l for l in Lr_i), vtype=GRB.BINARY, name="y")
    z = mPH2.addVars((l for l in Lr_i), vtype=GRB.CONTINUOUS, name="z")
    u = mPH2.addVars(((l, m) for l in Lr_i for m in Lr_i), vtype=GRB.BINARY, name="u")
    w = mPH2.addVars(((l, m) for l in Lr_i for m in Lr_i), vtype=GRB.CONTINUOUS, name="u")

    # Objective Function
    obj_expr = (
        gp.quicksum(L_i[l][4] * y[l] for l in Lr_i) + # Term [C3/C4]
        delta * gp.quicksum(E_i[l][m] * u[l, m] for l in Lr_i for m in Lr_i) # Term [C5]
    )
    mPH2.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints
    mPH2.addConstr(gp.quicksum(z[l] for l in LTTr_i) == Vi) # total supply
    mPH2.addConstr(gp.quicksum(z[l] for l in LFDr_i) == Vi) # total demand
    mPH2.addConstrs(EPS * y[l] <= z[l] for l in Lr_i)  # demand/supply iff facility is chosen
    mPH2.addConstrs(z[l] <= L_i[l][3] * y[l] for l in Lr_i) # demand/supply iff facility is chosen
    mPH2.addConstrs( # flow iff pipeline is built
        EPS * u[l, m] <= w[l, m] for l in Lr_i for m in Lr_i)  
    mPH2.addConstrs( # flow iff pipeline is built
        w[l, m] <= max(*G, F) * u[l, m] for l in Lr_i for m in Lr_i)  
       # w[l, m] <= min(L_i[l][3],L_i[m][3]) * u[l, m] for l in Lr_i for m in Lr_i)  
    mPH2.addConstrs( gp.quicksum(w[l, m] for m in Lr_i) # flow balance
                    - gp.quicksum(w[m, l] for m in Lr_i) == (1 - 2*L_i[l][2]) * z[l] for l in Lr_i)               
    for l in Lr_i: u[l, l].ub = 0 # no flow from/to same node
    mPH2.addConstrs(u[l, m] + u[m, l] <= 1 for l in Lr_i for m in Lr_i)  # demand/supply iff facility is chosen

    # min distance constraints
    for l in Lr_i:
        for m in Lr_i:
            if l != m:
                if (E_i[l][m] < D_MAT[L_i[l][5]][L_i[m][5]]):    
                    mPH2.addConstr(y[l] + y[m] <= 1)     

    # warm-start with Phase 1-2 solution
    if WARM == 1:
        for l in Lr_i:
            y[l].start = y_H[l]
            for m in Lr_i:
                u[l, m].start = u_H[l][m] 

    # Optimize the model
    mPH2.Params.MIPGap = MODEL_gap
    mPH2.Params.TimeLimit = MODEL_runtime
    mPH2.update()
    mPH2.optimize()

    # extract solution components
    obj = [] # first element full objective, then its components [C1]-[C5]
    obj.append(mPH2.objVal) 
    obj.append( sum(L_i[l][4] * y[l].x for l in LTTr_i) ) 
    obj.append( sum(L_i[l][4] * y[l].x for l in LFDr_i) )
    obj.append( delta * sum(E_i[l][m] * u[l, m].x for l in Lr_i for m in Lr_i) )

    yL, zL, uL, wL = [], [], [], []
    for l in Lr_i:
        yL.append(y[l].x)
        zL.append(z[l].x)
        uLsingle = []
        wLsingle = []
        for m in Lr_i:
            uLsingle.append(u[l, m].x)
            wLsingle.append(w[l, m].x)
        uL.append(uLsingle)
        wL.append(wLsingle)

    return obj,yL,zL,wL,uL,mPH2.MIPGap

# solve yard-based problem using a constructive heuristic based on configurations
def configuration_heuristic_v3(LTTr_i,LFDr_i,E_i,L_i,NCi,SCALE,D_MAT,MAX_FAC,delta):

    # select FDs from which to build configurations
    loc_FD = []
    loc_temp = []
    for l in LFDr_i:
        if L_i[l][5] == 2:
            loc_FD.append(l) # consider all slow FDs
        else:  
            loc_temp.append(l) # consider all fast FDs
    random.shuffle(loc_temp)
    cutoff = int(len(loc_temp) * SCALE)
    if cutoff < 10:
        cutoff = min(10, len(loc_temp)) # take anyway 10
    for l in range(0,cutoff):
        loc_FD.append(loc_temp[l])

    # construct pairs by finding best TT for each selected FD
    pairs = []
    for m in loc_FD: # selected FDs
        cost_best = 10 ** 9 # initialize with high number
        for l in LTTr_i: # all possible TTs
            if E_i[l][m] >= D_MAT[L_i[l][5]][L_i[m][5]]: # minumum distance
                cost_pair = L_i[l][4] + L_i[m][4]  + delta * E_i[l][m] 
                if cost_pair < cost_best:
                    capa_pair = min(L_i[l][3],L_i[m][3])
                    candidate = [l,m,cost_pair,cost_pair/capa_pair,L_i[l][3],L_i[m][3],capa_pair]
                    cost_best = cost_pair
        pairs.append(candidate)

    # extend with configurations
    config = []
    edges = []

    #for i in range(0,len(pairs)): # extend all pairs
    for i in range(0,len(pairs)):
        LOC1 , LOC2, COST, REL_COST, SUPPLY, DEMAND, CAPACITY = pairs[i] # add pair to facility
        # trains served, total cost, num locations, locations number
        NUM_FAC = 2 # number of facilities in the configuration
        curr_c = [CAPACITY, SUPPLY, DEMAND, COST, REL_COST, NUM_FAC, LOC1, LOC2] 
        config.append(curr_c.copy())
        edge = [LOC1,LOC2]
        edges_c=[]
        edges_c.append(edge)
        curr_e = edges_c
        edges.append(edges_c)

        while curr_c[5] < MAX_FAC: # expand current_config with one more facility
            best_cost = 10 ** 10 
            if curr_c[1] > curr_c[2]: L_cand, type = LFDr_i, 1 # consider FD locations
            if curr_c[2] > curr_c[1]: L_cand, type = LTTr_i, 0 # consider TT locations
            addition_possible = False
            for j in (x for x in L_cand if x not in curr_c[6:6+curr_c[5]]):
                # check feasibility and best connection
                feasible = True
                extra_cost = [0] * curr_c[5]
                for f in range(0,curr_c[5]): # check feasibility
                    if E_i[j][curr_c[6+f]] < D_MAT[L_i[j][5]][L_i[curr_c[6+f]][5]]: 
                        feasible = False
                        break
                    else: 
                        extra_cost[f] = delta * E_i[j][curr_c[6+f]] + L_i[j][4] 
                if feasible == True: # find best connection with existing facilities
                    addition_possible = True
                    min_cost = min(extra_cost)
                    min_index = extra_cost.index(min_cost)
                    if min_cost < best_cost: # update best candidate configuration and connection
                        best_cost = min_cost
                        candidate_c = curr_c.copy()
                        candidate_c.append(j) # add new element
                        if type == 0: candidate_c[1] += L_i[j][3] # add TT
                        if type == 1: candidate_c[2] += L_i[j][3] # add FD
                        candidate_c[0] = min(candidate_c[1],candidate_c[2])
                        candidate_c[3] += best_cost
                        candidate_c[4] = candidate_c[3] / candidate_c[0]
                        candidate_c[5] += 1
                        candidate_e = [j, curr_c[6+min_index]]

            if addition_possible == False: break 
            if addition_possible == True: # add best found to configuration set
                #print('FINAL:',candidate_c,', cost',best_cost)
                config.append(candidate_c.copy())
                curr_c = candidate_c.copy()
                edges_c = curr_e.copy()
                edges_c.append(candidate_e.copy())
                edges.append(edges_c.copy())
                curr_e = edges_c.copy()
    
    #print_2D_list(config,"Configurations (Capa, Supp, Dema, Cost, RelCost, #locations, locations)") 
    C = len(config)
    Cr = range(0,C)

    # construct overlapping indicator
    overlap = [[0 for _ in Cr] for _ in Cr]
    for c1 in Cr: # first configuration
        for c2 in range(c1+1,C): # second configuration
            for pos1 in range(6, 6 + config[c1][5]): # locations in first configuration 
                for pos2 in range(6, 6 + config[c2][5]): # locations in second configuration 
                    l1, l2 = config[c1][pos1], config[c2][pos2]
                    if l1 == l2 or E_i[l1][l2] < D_MAT[L_i[l1][5]][L_i[l2][5]]: 
                        overlap[c1][c2] = 1
                        break
                if overlap[c1][c2] == 1:
                    break

    # select configurations
    FLAG = 2 # all problems are solved to optimality
    cost_i, chosen_i = [], []
    print("Solving number of trains: ", end="")
    cost_i.append(0.0)
    chosen_i.append([-1])
    for n in range(1, NCi+1): 
        print(n, " ", end="")
        
        # select configuration subset using a mathematical program
        mCONF = gp.Model('model_CONF') 
        mCONF.setParam('OutputFlag', 0)
        x = mCONF.addVars(C, vtype=GRB.BINARY, name="x") # configuration selection
        mCONF.addConstrs(x[c1] + x[c2] <= 1 for c1 in Cr for c2 in range(c1+1,C) if overlap[c1][c2] == 1) # non-overlapping
        mCONF.addConstr(gp.quicksum(config[c][0] * x[c] for c in Cr) >= n) # total flow fulfilled
        mCONF.setObjective(gp.quicksum(config[c][3] * x[c] for c in Cr) , GRB.MINIMIZE)  # minimize cost
        mCONF.optimize()
        if mCONF.status != 2: # check if solved to optimality
            FLAG = mCONF.status
            break
        else:
            cost_i.append(mCONF.objVal) # extract objective
            chosen_in = []
            for c in Cr: # extract decision variables
                if x[c].x > 0.1:
                    chosen_in.append(c) 
            chosen_i.append(chosen_in)
    print()
    return cost_i, config, edges, chosen_i, FLAG

