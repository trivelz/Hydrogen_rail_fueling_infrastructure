# Designing an optimized fueling infrastructure for a hydrogen-powered railway system
# Author: Alessio Trivella

import os
import time
import random
import read_and_print as rap
import models_and_algo as algo
import support_functions as supp

if __name__=="__main__":
    os.system('cls') # clear terminal screen

    #============================================================
    #input parameters
    #============================================================

    # solving parameters
    MODEL_type = 1 # 0=full model, 1=matheuristic
    TUNING_EXPERIMENT = 0 # tune matheuristic (number of pairs and configurations; only if MODEL_type = 1)
    APPROX_EXPERIMENT = 0 # test cost approximation experiments with heuristic vs MIP (only if MODEL_type = 1)
    # full model
    MIP_gap_full = 0.0/100 # gap for full MIP
    MIP_time_full = 600 # runtime for full MIP
    # matheuristic
    Run_Ph3 = True # run improvement phase (True) or not (False)
    MAX_FAC = 5 # maximum number of facilities in a configuration (selected: 5)
    SCALE = 0.7 # multiplier used in the heuristic to define number of pairs (selected: 0.7)
    MIP_gap_Ph3 = 0.0/100 # gap for phase-3 MIPs
    MIP_time_Ph3 = 200 # runtime phase-2 MIPs
    WARM = 1 # warm start solution from previous phases
    random.seed(1)

    # discretization and plot options
    Balanced = 0 # compute a balanced disctretization of areas and lines
    STEP_TT = 10 # step [m] to discretize TT areas [for Balanced = 0]
    STEP_FFI = 10 # step [m] to discretize FFI lines [for Balanced = 0]
    POINTS = 200 # approximate number of points, >=50 [for Balanced = 1]
    PLOT_AREA = 0 # visualize discretized areas (1) or not (0)
    PLOT_LINE = 0 # visualize discretized lines (1) or not (0)

    # Models constraints
    D_MIN = [25,25,25,50,25,0] # min distance  TT-TT / TT-FFD / TT-SFD / FFD-FFD / FFD-SFD / SFD-SFD 

    # ============================================================
    # read and process input data
    # ============================================================

    D_MAT = matrix = [[D_MIN[0], D_MIN[1], D_MIN[2]], [D_MIN[1], D_MIN[3], D_MIN[4]], [D_MIN[2], D_MIN[4], D_MIN[5]]]

    instance_file_path = 'instances/baseline_shape.xlsx'
    instance_check_file_path = 'instances/baseline_shape_CHECK.txt'

    # read information from general instance sheet
    I,S,N,F,delta,K,rho,Ir,Sr,Nr,Kr,G,Sn,Ds,Dn =\
        rap.read_instance_general(instance_file_path)
    # read information from yard worksheets
    alpha,C,beta,gamma,items,A_lati,A_longi,L_lati,L_longi,P_lati,P_longi =\
        rap.read_instance_yard(instance_file_path,I)
    # discretize areas, lines, and return lists of all locations
    if Balanced == 0:
        J,Jr,lati_ij,longi_ij,H,Hr,lati_ikh,longi_ikh,Mi,Mk = supp.determine_locations(Ir,Kr,items,
            A_lati,A_longi,L_lati,L_longi,P_lati,P_longi,STEP_TT,STEP_FFI,PLOT_AREA,PLOT_LINE,C,F,G)
    elif Balanced == 1:
        J,Jr,lati_ij,longi_ij,H,Hr,lati_ikh,longi_ikh,Mi,Mk = supp.determine_locations_balanced(Ir,Kr,items,
            A_lati,A_longi,L_lati,L_longi,P_lati,P_longi,POINTS,PLOT_AREA,PLOT_LINE,C,F,G)
    # transform coordinates into distance matrices 
    E,L,Lr,LTTr,LFDr = supp.compute_distances(lati_ij,longi_ij,lati_ikh,longi_ikh,Ir,Jr,Kr,Hr,F,G,beta,gamma)

    # print instance to check correctness
    rap.print_instance_data(I,S,N,F,delta,K,rho,Ir,Sr,G,Sn,Ds,Dn,alpha,C,beta,gamma,
        items,A_lati,A_longi,L_lati,L_longi,P_lati,P_longi,instance_check_file_path)
    if Balanced == 0:
        name = 'instances/solution_' + str(MODEL_type) + '_STEP_' + str(STEP_TT) + '_' + str(STEP_FFI) + '_' + str(MIP_time_full) + 's' + '_' + str(N) + 'n'
    elif Balanced == 1:
        name = 'instances/solution_' + str(MODEL_type) + '_BAL_' + str(POINTS) + '_' + str(MIP_time_full) + 's'
    solution_file_path = [name + '_MIP.txt', name + '_MIP_MAT.m', name + '_HP2.txt', 
                    name + '_HP2_MAT.m', name + '_HP3.txt', name + '_HP3_MAT.m']

    # ============================================================
    # Solve model
    # ============================================================

    if MODEL_type == 0: # solve MIP using Gurobi
        x,v,y,z,w,u = algo.solveMIP(solution_file_path,I,S,Ir,Nr,Lr,LTTr,LFDr,E,L,Dn,alpha,rho,delta,Kr,Sn,C,F,G,D_MAT,MIP_gap_full,MIP_time_full)     

    elif MODEL_type == 1: # matheuristic
        if TUNING_EXPERIMENT == 1: supp.tuning_configurations_setting (LTTr,LFDr,E,L,N,C,Ir,D_MAT) # test different heuristic configurations
        
        start_time = time.time()
        print("-----------------------------------PHASE 1 -------------------------------------")
        print("Heuristic setting (MAX_FAC, SCALE) = (", MAX_FAC,",",SCALE,")")
        cost, config, edges, chosen = [], [], [], [] # per-yard cost and configurations from heuristic
        for i in Ir:
            cost_i, config_i, edges_i, chosen_i, FLAG = algo.configuration_heuristic_v3(LTTr[i],
                                                        LFDr[i],E[i],L[i],min(N,C[i]),SCALE,D_MAT,MAX_FAC,delta)
            cost.append(cost_i), config.append(config_i), edges.append(edges_i), chosen.append(chosen_i)
            if FLAG != 2: # check if all optimization problems in phase 1 were feasible
                print("Setting: (MAX_FAC,SCALE) = (", MAX_FAC,", ",SCALE,") --> INFEASIBLE (flag ",FLAG,")")
                break
        if FLAG == 2:
            print("Total cost all combination: ", sum(value for row in cost for value in row))
            print(" --> Time to run Phase 1: %.2fs\n" %(time.time() - start_time)) 
            #print_2D_list(cost,"Costs from phase 1\n") # show all costs for subproblems

        print("-----------------------------------PHASE 2-------------------------------------")
        obj1,x,v_sum,v = algo.MIP_phase1(I,Ir,Nr,C,cost,rho,alpha,Dn) # solve single-stage facility location
        runtime = time.time() - start_time
        print("Allocation: ", v_sum, ", objective: ", obj1)
        print(" --> Time to run Phases 1+2: %.2fs\n" %runtime) 
        y_H, z_H, u_H, w_H = supp.extract_solution_Ph1_Ph2(Lr,x,[36,12,12,0],config,edges,chosen,Ir)
        rap.print_solution_HEUR(solution_file_path[2],x,v,y_H,z_H,w_H,u_H,Ir,Kr,I,S,Lr,LTTr,LFDr,["n.a."] * len(Ir),runtime,
                                alpha,delta,rho,Dn,Nr,E,L,G,Sn)
        rap.print_solution_Matlab_HEUR(solution_file_path[3],x,v,y_H,z_H,w_H,u_H,Ir,I,L,Lr,LTTr,LFDr,obj1[0],
                                       alpha,delta,rho,Dn,Nr,E)
        if APPROX_EXPERIMENT == 1: # make experiments on cost approximation
            WARM = 0
            obj, gap, rtime, u_H, y_H = [], [], [], [], []
            with open("instances/RESULTS_APPROX_SAVE.txt", 'w') as file:
                print("Experiments on cost approximation heuristic vs MIP\n", file=file)
            for i in Ir:
                obj_i, gap_i, rtime_i = [], [], []
                obj_i.append(0), gap_i.append(0), rtime_i.append(0), 
                for n in range(1, min(N,C[i])+1):
                    start_time_exp = time.time()
                    res,y_opt,z_opt,w_opt,u_opt,gap_in = algo.MIP_phase2(Lr[i],LTTr[i],LFDr[i],
                                E[i],L[i],n,delta,F,G,D_MAT,MIP_gap_Ph3,MIP_time_Ph3,WARM,u_H,y_H) 
                    obj_i.append(res[0]), gap_i.append(gap_in), rtime_i.append(time.time() - start_time_exp) 
                    print("\nPhase 3: (i, n) = (", i,",",n,") --> (obj, gap, runtime) = (", obj_i[n],",",gap_i[n],",",rtime_i[n],")")
                    with open("instances/RESULTS_APPROX_SAVE.txt", 'r+') as file:
                        file.seek(0, 2)
                        print("Phase 3: (i, n) = (", i,",",n,") --> (obj, gap, runtime) = (", obj_i[n],",",gap_i[n],",",rtime_i[n],")", file=file)
                obj.append(obj_i), gap.append(gap_i), rtime.append(rtime_i)
                
            rap.print_2D_list(cost,"Costs from phase 1\n") # show all costs for subproblems
            rap.print_2D_list(obj,"Cost with the yard-level MIP\n") # show all costs for subproblems
            rap.print_2D_list(gap,"Gap of yard-level MIP\n") # show all costs for subproblems
            rap.print_2D_list(rtime,"Runtime of yard-level MIP\n") # show all costs for subproblems

        if Run_Ph3 == True:
            print("\n-----------------------------------PHASE 3-------------------------------------")
            obj2, y, z, w, u, gap = [], [], [], [], [], []
            for i in Ir:
                if x[i] > 0.5:
                    res,y_opt,z_opt,w_opt,u_opt,gap_i = algo.MIP_phase2(Lr[i],LTTr[i],LFDr[i],
                                E[i],L[i],v_sum[i],delta,F,G,D_MAT,MIP_gap_Ph3,MIP_time_Ph3,WARM,u_H[i],y_H[i])
                    obj2.append(res[0])
                    y.append(y_opt), z.append(z_opt), w.append(w_opt), u.append(u_opt), gap.append(gap_i)
                else:
                    obj2.append(0.0), y.append(-1), z.append(-1), w.append(-1), u.append(-1), gap.append(-1)

            obj2.append(obj1[1]), obj2.append(sum(obj2))
            runtime = time.time() - start_time
            rap.print_solution_HEUR(solution_file_path[4],x,v,y,z,w,u,Ir,Kr,I,S,Lr,LTTr,LFDr,gap,runtime,
                                    alpha,delta,rho,Dn,Nr,E,L,G,Sn) 
            rap.print_solution_Matlab_HEUR(solution_file_path[5],x,v,y,z,w,u,Ir,I,L,Lr,LTTr,LFDr,obj2[-1],
                                           alpha,delta,rho,Dn,Nr,E)

            print("-----------------------------------SUMMARY-------------------------------------")
            print("\nSolutions phase 2\n", obj1, x, v_sum)
            for i in Ir:
                print("Yard %d :" %i, end = "")   
                print(x[i], " ", v_sum[i], " --> cost", cost[i][int(v_sum[i])])
            print("\nSolutions phase 3") # second phase
            print("Yard objective and total", obj2, "\n")
