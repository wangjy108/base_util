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

class Align():
    def __init__(self, **args):
        self.SearchMolObj = args["SearchMolObj"]
        self.RefMolObj = args["RefMolObj"]
        self.method = args["method"]

    def align_by_crippen3D(self, rdmolobj_mol, rdmolobj_ref):
        crippen_contrib_mol = rdMolDescriptors._CalcCrippenContribs(rdmolobj_mol)
        crippen_contrib_ref = rdMolDescriptors._CalcCrippenContribs(rdmolobj_ref)
        crippen3D = rdMolAlign.GetCrippenO3A(rdmolobj_mol, rdmolobj_ref, \
                                             crippen_contrib_mol, crippen_contrib_ref)
        crippen3D.Align()
        
        return rdmolobj_mol
       
    
    def align_by_LSalignFlex(self, rdmolobj_mol, rdmolobj_ref):
        ## step1: should first gen mol2 file
        save_search = Chem.SDWriter("TEMP_search.sdf")
        save_ref = Chem.SDWriter("TEMP_ref.sdf")
        save_search.write(rdmolobj_mol)
        save_ref.write(rdmolobj_ref)
        save_search.close()
        save_ref.close()

        os.system("obabel -isdf TEMP_search.sdf -O TEMP_search.mol2 > /dev/null")
        os.system("obabel -isdf TEMP_ref.sdf -O TEMP_ref.mol2 > /dev/null")

        ## check if transformation status is ok
        if not (os.path.exists("TEMP_search.mol2") and os.path.exists("TEMP_ref.mol2")):
            logging.info("Fail to generate mol2 file for ref or/and search, use other methods instead")
        return

        ## step2: perform align
        os.system(f"LSalign TEMP_search.mol2 TEMP_ref.mol2 -rf 1 -o Aligned_search.pdb -acc 1")
        os.system(f"grep 'QUE 88888' Aligned_search.pdb > Aligned_search_1.pdb")
        os.system(f"obabel -ipdb Aligned_search_1.pdb -O Aligned_search_1.sdf")
        
        aligned_rdmolobj_search = [cc for cc in Chem.SDMolSupplier(f"Aligned_search_1.sdf", removeHs=False)][0]

        try:
            aligned_rdmolobj_search.SetProp("_Name", f"Aligned_mol")
        except Exception as e:
            rigid_align = self.align_by_crippen3D(mol_search, ref)
            rigid_align.SetProp("_Name", f"Aligned_mol_rig")
            aligned_rdmolobj_search = rigid_align
            
        os.system(f"rm -f Aligned_* TEMP_*")
        return aligned_rdmolobj_search

    def align_by_selfDesign():
        return 
    
    def run(self):
        run_dict = {"crippen3D": self.align_by_crippen3D, \
                    "LSalignFlex": self.align_by_LSalignFlex}
        try:
            run_dict[self.method]
        except Exception as e:
            print(f"Wrong method setting, please choose from [{[kk for kk in run_dict.keys()]}]")
            return None
        
        get_aligned_mol = run_dict[self.method](self.SearchMolObj, self.RefMolObj)

        return get_aligned_mol


