# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 01:56:41 2022

@author: mahza
"""

from gurobipy import *
import numpy as np

def optimize(NUM_VNFS, VNF_TYPE,
 NUM_HOSTS, HOST_TYPE, MATRIX,
  HOST_CAPACITY, VNF_TOLERANCE, VNF_REQUIRE,
   HOST_LINK, VNF_BANDWIDTH, verbose=True, time_limit=False, obj1_weight=1.0, obj2_weight=1.0):
    
    #Ct = host_capacity
    #Wt = host link
    #wri = vnf bandwidth
    #tri = matrix
    #nri = number of vnfs
    #cf = vnf requirement
    #new thing = vnf tolerance

    m = Model("vnf")
   
    if verbose == False:
        m.setParam("OutputFlag", 0) # do not show the output log
        
    if time_limit != False:
        m.setParam('TimeLimit', time_limit)
   
    
    """ binary solution (variables) """
    solution = []
    for i in range(NUM_VNFS):
        tmp = []
        for j in range(NUM_HOSTS):
            x = m.addVar(vtype=GRB.BINARY, name="vnf"+str(i)+"_host"+str(j))
            tmp.append(x)
        solution.append(tmp)
    
    solution=np.array(solution)
    """ optimization objective 1 (minimize latency) """
    obj_function = 0
    for i in range(NUM_VNFS):
        for j in range(NUM_HOSTS):
            obj_function += (MATRIX[i,j] * solution[i][j])
    m.setObjectiveN(obj_function,0,0,obj1_weight)
    
    """ optimization objective 2 (maximize VNFs) """
    obj_function_2 = 0
    for i in range(NUM_VNFS):
        for j in range(NUM_HOSTS):
            obj_function_2 += solution[i,j]
    m.setObjectiveN(NUM_VNFS-obj_function_2,1,1, obj2_weight)
    
    m.ModelSense = GRB.MINIMIZE

    """ constraint 1 (latency tolerance) """
    for i in range(NUM_VNFS):
        latency = 0
        for j in range(NUM_HOSTS):
            latency += (MATRIX[i,j] * solution[i][j])
        m.addConstr(latency <= VNF_TOLERANCE[i])

    """ constraint 2 (host capacity) """
    for j in range(NUM_HOSTS):
        capacity_required = 0
        for i in range(NUM_VNFS):
            capacity_required += (VNF_REQUIRE[i] * solution[i][j])
        m.addConstr(capacity_required <= HOST_CAPACITY[j])
    
    #MAIN CAPACITY
    """ constraint 3 (host link bandwidth capacity) """
    for j in range(NUM_HOSTS):
        capacity_required = 0
        for i in range(NUM_VNFS):
            capacity_required += (VNF_BANDWIDTH[i] * solution[i][j])
        m.addConstr(capacity_required <= HOST_LINK[j])
        
    """ constraint 4 (ensure each VNF has and only has one matched host) """
    for i in range(NUM_VNFS):
        tmp = 0
        for j in range(NUM_HOSTS):
            tmp += solution[i][j]
        m.addConstr(tmp <= 1)
        
    """ constraint 5 """
    for i in range(NUM_VNFS):
        for j in range(NUM_HOSTS):
            m.addConstr(VNF_TYPE[i]*solution[i][j] <= HOST_TYPE[j])
            
    """ optimize model """
    m.optimize()
    
    solution_ = np.zeros([NUM_VNFS, NUM_HOSTS], np.int32) 
    for i in range(NUM_VNFS):
        for j in range(NUM_HOSTS):
            solution_[i][j] = int(solution[i][j].x)
    
    print(solution)
    print(m.objVal)
    n_placed_vnfs = np.reshape(np.copy(solution_), [-1]).sum()
    
    return solution_, m.objVal/n_placed_vnfs, n_placed_vnfs

sol = optimize(2, np.array([1,1]), 2, np.array([1,2]), np.array([[1,2],[2,1]]), [10,20], [20,20], [1,1], [10,10], [1,1], verbose=True, time_limit=False, obj1_weight=1.0, obj2_weight=1)