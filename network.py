#!/usr/bin/python3
# coding: utf-8

import multiprocessing as mp
import numpy as np
import osmnx as ox


"""
Simplified road network 

"""
class network:
   
    def __init__(self):
        pass

    def parse(self, south , north, west, east):
        """
        return simplified road network
        """
        G = ox.graph_from_bbox(south , north, west, east , network_type='drive',simplify=False,truncate_by_edge=True)
        gdf_nodes, gdf_edges=ox.graph_to_gdfs(G)
        del_list=[]
        for j,i in gdf_edges.iterrows():
            if i["highway"]=='living_street' or i["highway"]=='residential' or i["highway"]=='unclassified' \
            or i["highway"]=="motorway_link":
                
                del_list.append(j)

        gdf_edges=gdf_edges.drop(index=del_list)
        G_sim=ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None)  
        return G_sim      

    def match_network(self, G):
        """
        save matched road network
        """
        _,gdf_edges = ox.graph_to_gdfs(G)
        gdf_edges.index.names=['source','target','key']
        gdf_edges=gdf_edges.reset_index()
        road_network=gdf_edges[['source', 'target', 'length', 'osmid', 'highway',
       'geometry']]
        road_network['_uid_']=np.arange(1,len(road_network)+1)
        road_network['id']=np.arange(1,len(road_network)+1)
        road_network=road_network.rename({'length':'cost'},axis=1)
        road_network.cost=road_network.geometry.length
        road_network.highway.unique()
        #road_network[~road_network.highway.isin(['disused','residential','living_street','motorway_link'])].plot()
        road_network[['source', 'target', 'cost', 'osmid', 'highway', '_uid_', 'id',
            'geometry']].to_file('/data/network.shp')
