import graphity
import numpy as np
import networkx as nx
import r2pipe
from joblib import load, dump
from malwareDetector.detector import detector
from utils import parameter_parser
from utils import write_output
import os
import time

class GraphTheoryDetector(detector):
    def __init__(self, model_name_or_path:str='', task_type:str='detection', training:bool=False) -> None:
        self.model_name_or_path = model_name_or_path
        self.task_type = task_type

        if training:
            self.model = None
            return
        
        if task_type != 'detection' and task_type != 'classification':
            raise ValueError('task_type must be either detection or classification')

        # load model
        self.model = None
        if '/' in model_name_or_path or os.path.exists(model_name_or_path):
            # specify path to the model
            self.model = load(self.model_name_or_path)
        else:
            # use pre-train model
            dir_path = os.path.dirname(__file__)
            if task_type == 'detection':
                dir_path = os.path.join(dir_path, 'MD_Model')
            else:
                dir_path = os.path.join(dir_path, 'FC_Model')

            self.model = load(f'{dir_path}/{model_name_or_path}.joblib')
            
    def extractFeature(self, fpath:str) -> nx.graph:
        '''
        Extract features from a binary file using the radare2 library.
        
        Args:
            fpath (str): The path to the file to be processed.

        Returns:
            nx.graph: A NetworkX graph object representing the file's FCG.
        '''
        r2 = r2pipe.open(fpath)
        r2.cmd('aaaa')
        fcg_dot_string = r2.cmd('agCd')
        fcg = graphity.create_graph(fcg_dot_string)
        return fcg
    
    def vectorize(self, fcg:nx.graph) -> np.array:
        '''
        Extract feature vector from FCG based on graph theory

        Args:
        Returns:
        '''
        feature=[]

        # append #nodes & # edges
        feature.append(fcg.number_of_nodes())
        feature.append(fcg.number_of_edges())

        # append Density
        feature.append(graphity.get_density(fcg))

        # append Closeness Centrality
        for i in graphity.closeness_centrality(fcg):
            feature.append(i)

        # append Betweeness Centrality
        for i in graphity.betweeness_centrality(fcg):
            feature.append(i)

        # append Degree Centrality
        for i in graphity.degree_centrality(fcg):
            feature.append(i)

        # append Shortest Path
        for i in graphity.shortest_path(fcg):
            feature.append(i)

        # scaling
        scaler = load('scaler.joblib')
        feature = scaler.transform(np.array(feature).reshape(1,-1))

        return np.array(feature)

    def predict(self, feature_vector:np.array) -> np.array:
        return self.model.predict_proba(feature_vector).tolist()[0]

    def predict_raw_feature(self, raw_feature:nx.graph) -> np.array:
        feature_vector = self.vectorize(raw_feature)
        return self.predict(feature_vector)

    def predict_binary_file(self, fpath:str) -> np.array:
        raw_feature = self.extractFeature(fpath)
        feature_vector = self.vectorize(raw_feature)
        return self.predict(feature_vector)
    
    def train(self, feature:np.array, label:np.array, model:object) -> None:
        print('Start trianing')
        start = time.time()
        self.model = model.fit(feature, label)
        end = time.time()
        print(f'Time cost: {end-start} sec')

    def save_model(self, fpath:str) -> None:
        dump(self.model, fpath)