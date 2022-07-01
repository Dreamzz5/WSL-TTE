#!/usr/bin/python3
# coding: utf-8

from fmm import Network,NetworkGraph,FastMapMatch,FastMapMatchConfig,UBODT,UBODTGenAlgorithm
from fmm import GPSConfig,ResultConfig
import glob
from tqdm import tqdm
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import os 
from shapely import wkt
import pickle as pk

class matching:
   
    def __init__(self):
        pass

    def matching(self, traj_path,network_path):
        k = 8
        radius = 300/1.132e5
        gps_error =50/1.132e5
        input_config = GPSConfig()
        input_config.id = "order_id"
        input_config.timestamp='timestamp'
        input_config.y='latitude'
        input_config.x='longitude'

        result_config=ResultConfig()
        result_config.output_config.write_cpath=True
        result_config.output_config.write_duration=True
        result_config.output_config.write_ep=True
        result_config.output_config.write_error=True
        result_config.output_config.write_length=True
        result_config.output_config.write_mgeom=True
        result_config.output_config.write_offset=True
        result_config.output_config.write_opath=True
        result_config.output_config.write_pgeom=True
        result_config.output_config.write_spdist=True
        result_config.output_config.write_speed=True
        result_config.output_config.write_tp=True
        result_config.output_config.write_tpath=True

        fmm_config = FastMapMatchConfig(k,radius,gps_error)
        network = Network(network_path)
        print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count())) 
        graph = NetworkGraph(network)   
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        status = ubodt_gen.generate_ubodt("ubodt.txt", 0.02, binary=False, use_omp=True)
        print (status)
        ubodt = UBODT.read_ubodt_csv("ubodt.txt")
        model = FastMapMatch(network,graph,ubodt)


        traj_dict={"order_id":[],"timestamp":[],"longitude":[],"latitude":[]}
        trajs=pk.load(open(traj_path,"rb"))

        for idx,traj in enumerate(trajs):
            for temp in traj["trajs"]:
                traj_dict["order_id"].append(idx)
                traj_dict["timestamp"].append(temp[0])
                traj_dict["longitude"].append(temp[2])
                traj_dict["latitude"].append(temp[1])
        df=pd.DataFrame(traj_dict)
        df.to_csv("./FMM/gps.csv",index=None,sep=';')
        for file in tqdm(glob.glob('FMM/*')):
   
            input_config.file = file
            input_config.gps_point=True
            result_config.file = "gps.txt"
            status = model.match_gps_file(input_config, result_config, fmm_config)
            print(status)
            result=pd.read_csv("gps.txt",delimiter=";").dropna()
            return result
