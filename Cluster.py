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
from Align import *
from CalRMSD import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class cluster():
    def __init__(self, **args):
        self.sdf_file = args["inputSDF_fileName"]
        self.save_n = args["save_n"]

        self.mol_set = [mm for mm in Chem.SDMolSupplier(self.sdf_file, removeHs=False) if mm != None]
        ## extrac file Name
        self.mol_name = self.sdf_file.strip().split("/")[-1]

        self.avail_prop = [nn for nn in self.mol_set[0].GetPropNames()]

        try:
            self.method = args["method"]
        except Exception as e:
            self.method = 'RMSD'
        else:
            if self.method != 'RMSD':
                try:
                    self.mol_set[0].GetProp(self.method)
                except Exception as e:
                    logging.info(f"Not valid method, use RMSD instead")
                    self.method = 'RMSD'
            
        
        try:
            self.tag = args["name_tag"]
        except Exception as e:
            self.tag = "_Name"
        
        try:
            self.align_method = args["align_method"]
        except Exception as e:
            self.align_method = "crippen3D"

        try:
            self.rmsd_method = args["rmsd_method"]
        except Exception as e:
            self.rmsd_method = "selfWhole" 
        
        try:
            self.setReference = args["reference"]
        except Exception as e:
            self.setReference = 0
        else:
            try:
                self.reference = [mm for mm in Chem.SDMolSupplier(self.setReference) if mm][0]
            except Exception as e:
                self.reference = None

        self._path = os.getcwd()           
            
    
    def run(self):
        try:
            self.mol_set[0].GetProp(self.tag)
        except Exception as e:
            logging.info(f"Not valid property tag, current available property tags are: {self.avail_prop}")
            logging.info("choose one from these properties and run again")
            return 
        
        mol_collect = {}
        for i in range(len(self.mol_set)):
            constant_name = self.mol_set[i].GetProp(self.tag)
            if not constant_name in mol_collect.keys():
                mol_collect.setdefault(constant_name, [])
            mol_collect[constant_name].append(self.mol_set[i])

        save = []

        for k, v in mol_collect.items():
            logging.info(f"Working with {k}")
            if self.setReference and self.reference:
                aligned_v = [Align(SearchMolObj=v[ii], RefMolObj=self.reference, method=self.align_method).run() for ii in range(len(v))]
                sorted_by_rmsd = sorted([(RMSD(rdmolobj_mol=aligned_v[i], rdmolobj_ref=self.reference, method=self.rmsd_method).run(), aligned_v[i]) \
                                for i in range(len(aligned_v))], key=lambda x:x[0])
            else:
                aligned_v = [Align(SearchMolObj=v[ii], RefMolObj=v[0], method=self.align_method).run() for ii in range(len(v))]
                sorted_by_rmsd = sorted([(RMSD(rdmolobj_mol=aligned_v[i], rdmolobj_ref=aligned_v[0], method=self.rmsd_method).run(), aligned_v[i]) \
                                    for i in range(len(aligned_v))], key=lambda x:x[0])
            if len(v) > self.save_n and self.save_n > 1:
                #rmsd_mol = sorted([align_by_3Dcrippen(v[i], v[0]) for i in range(len(v))], key=lambda x:x[0])
                rmsd = [cc[0] for cc in sorted_by_rmsd]
                #sorted(rmsd_mol, key=lambda x:x[0])
                bins = np.linspace(min(rmsd), max(rmsd)+10e-6, self.save_n+1)
                if bins[-1] > 1e10:
                    bins[-1] = math.inf
                
                collect = {}
                for i in range(self.save_n):
                    collect.setdefault(i, [])

                collect[0].append(sorted_by_rmsd[0][1])
                #collect[n_cluster-1].append(rmsd_mol[-1][1])

                i = 1
                j = 1
                while i <  len(v):
                    item = sorted_by_rmsd[i]
                    if item[0] < bins[j]:
                        collect[j-1].append(item[1])
                    else:
                        j += 1
                        collect[j-1].append(item[1])
                    i += 1

               
                col_img = copy.deepcopy(collect)
                for kk, vv in col_img.items():
                    if len(vv) == 0:
                        del collect[kk]

                re_collect = sorted(list(collect.values()), key=lambda x:len(x), reverse=True)

                for cc in re_collect:
                    if self.method != "RMSD":
                        cc.sort(key=lambda x:float(x.GetProp(self.method)), reverse=True)
                    else:
                        cc = cc[::-1]

                current_save = []

                while True:
                    sel_from_which = self.save_n - len(current_save)
                    if sel_from_which > 0:
                        for ccc in re_collect[:sel_from_which]:
                            if ccc:
                                current_save.append(ccc.pop())
                    else:
                        break
                save += current_save
                #logging.info(f"collect as {[cc.GetProp('E_tot') for cc in current_save]}")

            else:
                save += [cc[-1] for cc in sorted_by_rmsd][:self.save_n]
                #print(rmsd_mol)
                    

        ww = Chem.SDWriter(os.path.join(self._path, "FILTER.sdf"))
        for cc in save:
            #cSMILES = pairwise_IdTage_smi[cc.GetProp("_Name")]
            #cc.SetProp("cSMILES", cSMILES)
            ww.write(cc)
        ww.close()

        logging.info(f"Save in need molecules in FILTER.sdf in {self._path}")
        return

    
