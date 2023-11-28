#!/usr/bin/env python
# coding: utf-8

"""
// author: Wang (Max) Jiayue 
// email: wangjy108@outlook.com
// git profile: https://github.com/wangjy108
"""

from rdkit import rdBase, Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import numpy as np
import os
import argparse
import copy
import scipy.spatial
import logging
import math
from collections import Counter, defaultdict
from util.Align import Align
#from util.CalRMSD import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class cluster():
    def __init__(self, **args):

        try:
            sample_mol_file_name = args["input_sdf"]
        except Exception as e:
            self.sample = None
        else:
            try:
                self.sample = [cc for cc in Chem.SDMolSupplier(sample_mol_file_name, removeHs=False) if cc]
            except Exception as e:
                self.sample = None
        
        if not self.sample:
            try:
                self.sample = args["input_rdmol_obj"]
            except Exception as e:
                self.sample = None
        
        try:
            self.do_align = args["do_align"]
        except Exception as e:
            self.do_align = False
        
        try:
            self.k = args["cluster_n"]
        except Exception as e:
            self.k = None
        else:
            if not isinstance(self.k, int):
                self.k = None

        try:
            self.distance_cutoff = args["rmsd_cutoff_cluster"]
        except Exception as e:
            self.distance_cutoff = 1.5
        else:
            if not isinstance(self.distance_cutoff, float):
                self.distance_cutoff = 1.5
        
        try:
            self.writer = args["if_write_cluster_mol"]
        except Exception as e:
            self.writer = False

        
    def get_xyz(self, mol):
        xyz = mol.GetConformer().GetPositions()
        heavy_atom_fac = np.array([1 if atom.GetSymbol() != 'H' else 0 for atom in mol.GetAtoms()]).reshape(-1, 1)
        xyz_heavy = np.multiply(xyz, heavy_atom_fac)
        _xyz = np.delete(xyz_heavy, np.where(np.sum(xyz_heavy, axis=1)==0.0)[0], axis=0)
        return _xyz
    
    def distance(self, ref_xyz, mol_xyz):
        naive_rmsd = np.power(sum((np.power(np.sum((mol_xyz - ref_xyz)**2, axis=1), 0.5))**2)/mol_xyz.shape[0], 0.5)
        return naive_rmsd
        
    def cluster(self):
        if not self.sample:
            logging.info("Error for input sampled mol file, terminated")
            return None
            
        #ref_xyz = self.get_xyz(self.ref)
        
        sample_xyz = [self.get_xyz(self.sample[i]) for i in range(len(self.sample))]

        _list_saved_mol = [self.sample[0]]
        _dic_saved_distance = {}

        ## decrease redundancy
        i = 1
        while i < len(self.sample):
            if self.do_align:
                involved_saved_mol = [Align(SearchMolObj=_list_saved_mol[ii],
                                            RefMolObj=self.sample[i],
                                            method="crippen3D").run()
                                        for ii in range(len(_list_saved_mol))]
            else:
                involved_saved_mol = _list_saved_mol
            
            distance_compare = [self.distance(self.get_xyz(involved_saved_mol[ii]), sample_xyz[i]) 
                                for ii in range(len(involved_saved_mol))]
            if min([cc for cc in distance_compare if cc > 0]) > self.distance_cutoff:
                _list_saved_mol.append(self.sample[i])
                tag = len(_list_saved_mol) - 1
                _dic_saved_distance.setdefault(tag, distance_compare)

            i += 1
            
        ## cluster

        if self.k and len(_list_saved_mol) <= self.k:
            return _list_saved_mol
        
        ## group by distance between mol and input ref
        distance_matrix = np.zeros((len(_list_saved_mol), len(_list_saved_mol)))
        for i in range(len(_list_saved_mol)):
            distance_matrix[i][i] = 999.99
            j = 0
            while j < i:
                distance_matrix[i][j] = distance_matrix[j][i] = _dic_saved_distance[i][j]
                j += 1
            
        min_idx = np.argmin(distance_matrix, axis=1)
        _cluster = defaultdict(int)
        for idx, _min in enumerate(min_idx):
            _cluster[idx] = _min

        idx_count = Counter(min_idx)

        clstr = defaultdict(list)

        tracker = []
        
        for _pair in [ii for ii in idx_count.most_common() if ii[1] > 1]:
            clstr[_pair[0]] = [kk for kk, vv in _cluster.items() if vv == _pair[0]]
            tracker.append(_pair[0])
            tracker += clstr[_pair[0]]

        rare_events = []

        for ii in range(len(min_idx)):
            if ii not in tracker:
                tag = _cluster[ii]
                get_items = [(kk, vv) for kk, vv in clstr.items() if tag in vv]

                if get_items:
                    del clstr[get_items[0][0]]
                    current_set = [get_items[0][0]] + get_items[0][1] + [ii]
                    this_matrix = np.zeros((len(current_set), len(current_set)))
                    for i in range(len(current_set)):
                        this_matrix[i][i] = 0.0
                        j = 0
                        while j < i:
                            this_matrix[i][j] = this_matrix[j][i] = distance_matrix[current_set[i]][current_set[j]]
                            j += 1
                    center = np.argmin(np.sum(this_matrix, axis=1))
                    clstr[current_set[center]] = [nn for nn in current_set if nn != current_set[center]]
                else:
                    rare_events += [ii, tag]
                    rare_events = list(set(rare_events))
            
            saved = [kk for kk, vv in clstr.items()] + rare_events
                    
            saved_mol = [_list_saved_mol[ii] for ii in set(saved)]
            
            return saved_mol
    
    def run(self):
        collect = self.cluster()

        if not collect:
            logging.info("Error when save clustered mol, terminated")
            return None
        
        if self.writer:
            logging.info("Save each cluster center mol in FILTER_*.sdf")
            for idx, mm in enumerate(collect):
                cc = Chem.SDWriter(f"FILTER_{idx}.sdf")
                cc.write(mm)
                cc.close()

        return collect










    
