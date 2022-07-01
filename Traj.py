#!/usr/bin/python3
# coding: utf-8

from sklearn.neighbors import KDTree
import pickle as pk
import networkx as nx
from itertools import islice
from tqdm import tqdm

def k_shortest_paths_nx(G, source, target, k, weight="length"):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

class Traj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """
    def __init__(self):
        pass

    def sampling(self, input_path,match_result,interval=60*4):
        """
        The sampling() function is to conduct low-sampling for original trajectories.
        """
        trajs=pk.load(open(input_path,"rb"))
        sampling_list=[]
        for traj_id,traj in enumerate(trajs):
    
            truth=match_result[match_result.id==traj_id]["cpath"].values[0].split(",")
           
            opath=match_result[match_result.id==traj_id]["opath"].values[0].split(",")
            offset=match_result[match_result.id==traj_id]["offset"].values[0].split(",")
            offset=list(map(lambda x:float(x), offset))
            temp=[]
            temp.append(traj["trajs"][0]+[opath[0]])
            time_start=traj["trajs"][0][0]
            idx=1
            for i in range(1,len(traj["trajs"])):
                if traj["trajs"][i][0]-time_start>=idx*interval:
                    temp.append(traj["trajs"][i]+[opath[i]])
                    idx+=1
            temp.append(traj["trajs"][-1]+[opath[-1]])
            sampling_list.append([truth,temp,[offset[0],offset[-1]]])
            return sampling_list
    def initial_path(self,G,sampling_list):
        """
        The initial_path() function is to generate intial paths for weak supervision.
        """
        gdf_nodes,gdf_edges = ox.graph_to_gdfs(G)
        gdf_edges["osmid"]=list(map(lambda x:str(x),list(range(1,len(gdf_edges)+1))))
        gdf_edges["length"]=gdf_edges.geometry.length      
        G_copy=ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None)
        nodes,edges = ox.graph_to_gdfs(G_copy)     
        G_dig=nx.DiGraph(G_copy)    
        edge_dict={}
        for i,j in edges.iterrows():
            edge_dict[i[0:2]]=j["osmid"]
        path_list=[]
        for j in tqdm(sampling_list):
            traj=j[1]
            is_keep=True
            interval_list=[]
            for i in range(1,len(traj)):
                
                if traj[i][-1]!=traj[i-1][-1]:
                    temp=list(edges[edges.osmid==traj[i-1][-1]].index)
                    temp=list(map(lambda x:x[0],temp))+list(map(lambda x:x[1],temp))
                    anchor_points=[]
                    
                    for node in temp:
                        anchor_points.append((nodes[nodes.index==node]["y"].values[0],nodes[nodes.index==node]["x"].values[0]))
                    anchor_points=np.array(anchor_points)
                    anchor_points = KDTree(anchor_points)
                    onid = temp[anchor_points.query([[ traj[i-1][1], traj[i-1][2]]], return_distance=False)[0, 0]]
                    
                    
                    temp=list(edges[edges.osmid==traj[i][-1]].index)
                    temp=list(map(lambda x:x[0],temp))+list(map(lambda x:x[1],temp))
                    anchor_points=[]
                    for node in temp:
                        anchor_points.append((nodes[nodes.index==node]["y"].values[0],nodes[nodes.index==node]["x"].values[0]))
                    anchor_points=np.array(anchor_points)
                    anchor_points = KDTree(anchor_points)
                    dnid = temp[anchor_points.query([[ traj[i][1], traj[i][2]]], return_distance=False)[0, 0]]
                    
                    if onid!=dnid:
                        try:
                            route=k_shortest_paths_nx(G_dig, onid, dnid,1, 'length')[0]
                            edge_list=[]
                            for k in range(1,len(route)):
                                edge_list.append(edge_dict[(route[k-1],route[k])])
                               
                        except Exception as error:
                            print(error)
                            is_keep=False
                    else:
                        is_keep=False
                    interval_list.append((i,edge_list))    
                    if not is_keep:
                        break 
            if is_keep:
                path_list.append((j,interval_list))
        return path_list