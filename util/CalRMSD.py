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

class RMSD():
    def __init__(self, **args):
        self.rdmolobj_mol = args["rdmolobj_mol"]
        self.rdmolobj_ref = args["rdmolobj_ref"]
        self.method = args["method"]

    def rmsd_self_whole(self, rdmolobj_mol, rdmolobj_ref):
        rdmolobj_mol = Chem.RemoveHs(rdmolobj_mol)
        rdmolobj_ref = Chem.RemoveHs(rdmolobj_ref)

        mol_xyz = rdmolobj_mol.GetConformer().GetPositions()
        ref_xyz = rdmolobj_ref.GetConformer().GetPositions()

        #hfac_mol = np.array([1 if atom.GetSymbol() != 'H' else 0 for atom in rdmolobj_mol.GetAtoms()]).reshape(-1, 1)
        #hfac_ref = np.array([1 if atom.GetSymbol() != 'H' else 0 for atom in rdmolobj_ref.GetAtoms()]).reshape(-1, 1)

        #mol_xyz = np.multiply(pre_mol_xyz, hfac_mol)
        #ref_xyz = np.multiply(pre_ref_xyz, hfac_ref)
        
        mol_atomIdx = np.array([a.GetAtomicNum() for a in rdmolobj_mol.GetAtoms()])
        ref_atomIdx = np.array([a.GetAtomicNum() for a in rdmolobj_ref.GetAtoms()])
        
        if mol_xyz.shape[0] > ref_xyz.shape[0]:
            search = ref_xyz
            search_atomIdx = ref_atomIdx
            target = mol_xyz
            target_atomIdx = mol_atomIdx
        else:
            search = mol_xyz
            search_atomIdx = mol_atomIdx
            target = ref_xyz
            target_atomIdx = ref_atomIdx
        
        match_target = []
        match_atomIdx = []
        
        for i in range(search.shape[0]):
            this_xyz = search[i]
            dis = scipy.spatial.distance.cdist(this_xyz.reshape(1, -1), target)
            minP = np.argmin(dis)
            match_target.append(target[minP])
            match_atomIdx.append(target_atomIdx[minP])
            np.delete(target, minP, axis=0)
            np.delete(target_atomIdx, minP)
        
        np_match_target = np.array(match_target).reshape(search.shape[0], 3)
        np_match_atomIdx = np.array(match_atomIdx)
        ## calc naive rmsd
        #naive_rmsd = np.power(sum((np.power(np.sum((search - np_match_target)**2, axis=1), 0.5))**2)/search.shape[0], 0.5)
        naive_rmsd = np.power(sum((np.power(np.sum((search - np_match_target)**2, axis=1), 0.5) \
                            * np.exp(np_match_atomIdx - search_atomIdx))**2)/search.shape[0], 0.5)
        
        return naive_rmsd
    
    def rmsd_rdkit_crippen3D(self, rdmolobj_mol, rdmolobj_ref):
        crippen_contrib_mol = rdMolDescriptors._CalcCrippenContribs(rdmolobj_mol)
        crippen_contrib_ref = rdMolDescriptors._CalcCrippenContribs(rdmolobj_ref)

        crippen3D = rdMolAlign.GetCrippenO3A(rdmolobj_mol, rdmolobj_ref,crippen_contrib_mol, crippen_contrib_ref)
        #crippen3D.Align()
        _map = crippen3D.Matches()
        rmsd = rdMolAlign.AlignMol(rdmolobj_mol, rdmolobj_ref, atomMap=_map)
        #rmsd = rmsd_self_whole(rdmolobj_mol, rdmolobj_ref)

        return rmsd
    
    def run(self):
        run_dict = {"crippen3D": self.rmsd_rdkit_crippen3D, \
                    "selfWhole": self.rmsd_self_whole}
        try:
            run_dict[self.method]
        except Exception as e:
            print(f"Wrong method setting, please choose from [{[kk for kk in run_dict.keys()]}]")
            return None
        
        get_rmsd = run_dict[self.method](self.rdmolobj_mol, self.rdmolobj_ref)

        return get_rmsd