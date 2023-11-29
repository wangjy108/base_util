#!/usr/bin/env python
# coding: utf-8

"""
// author: Jia Mengfei
// email: jmf43075@gmail.com
"""

from rdkit import rdBase, Chem
from rdkit.Chem import Draw, AllChem
import pandas as pd
import numpy as np
import os
import logging
import math
from joblib import Parallel, delayed
from pyscf import gto, dft, solvent
from pyscf.dftd3 import dftd3


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
// input sdf mol set, collect lowest SP energy and corresponding conformer
"""

class System():
    def __init__(self, **args):
        try:
            self.mol = [mm for mm in Chem.SDMolSupplier(args["input_sdf"], removeHs=False) if mm]
        except Exception as e:
            logging.info("Wrong input, check and run again")
            self.mol = None
        
        if not self.mol:
            try:
                self.mol = args["input_rdmol_obj"]
            except Exception as e:
                self.mol = None
        
        #self.method = args["method"]
        try:
            self.verbose = args["verbose"]
        except Exception as e:
            self.verbose = 0
        else:
            try:
                self.verbose = int(self.verbose)
            except Exception as e:
                self.verbose = 0
            else:
                if self.verbose > 9:
                    self.verbose = 9
        
        try:
            self.functional = args["functional"]
        except Exception as e:
            self.functional = 'b3lyp'
        
        try:
            self.charge_method = args["charge_method"]
        except Exception as e:
            self.charge_method = "auto"
        else:
            if not self.charge_method in ["auto", "read", "define"]:
                self.charge_method = "auto"
            
        if self.charge_method == "define":
            try:
                self.charge = int(args["define_charge"])
            except Exception as e:
                self.charge_method = "auto"
            else:
                if not isinstance(self.charge, int):
                    self.charge_method = "auto"
        
        try:
            self.basis = args["basis"]
        except Exception as e:
            self.basis = '6-31G*'
        
        try:
            self.if_D3 = args["if_D3"]
        except Exception as e:
            self.if_D3 = True
        else:
            if not isinstance(self.if_D3, bool):
                self.if_D3 = True
        
        try:
            self.cpu_core = args["cpu_core"]
        except Exception as e:
            self.cpu_core = 16
        else:
            try:
                self.cpu_core = int(self.cpu_core)
            except Exception as e:
                self.cpu_core = 16
        
        try:
            self.if_solvent = args["if_solvent"]
        except Exception as e:
            self.if_solvent = False
        else:
            if not isinstance(self.if_solvent, bool):
                self.if_solvent = False
        
        self.xyz_blocks = []
        
        for ii in range(len(self.mol)):
            #if not os.path.exists(os.path.join(self.workdir, f"run_{ii}")):
            #    os.mkdir(os.path.join(self.workdir, f"run_{ii}"))

            atom = [atom.GetSymbol() for atom in self.mol[ii].GetAtoms()]
            xyz = self.mol[ii].GetConformer().GetPositions()

            if self.charge_method == "define":
                charge = self.charge
            
            elif self.charge_method == "read":
                charge = int(self.mol[ii].GetProp("charge"))
            
            else:
                ## get charge
                AllChem.ComputeGasteigerCharges(self.mol[ii])
                _charge = sum([float(atom.GetProp("_GasteigerCharge")) for atom in self.mol[ii].GetAtoms()])

                ## sign
                if _charge:
                    charge_sign = _charge / abs(_charge)

                    if math.ceil(abs(_charge)) - abs(_charge) < 5e-1:
                        charge = math.ceil(abs(_charge)) * charge_sign
                    else:
                        charge = (math.ceil(abs(_charge)) - 1)* charge_sign
                else:
                    charge = int(_charge)

            ## save _input.xyz 
            df = pd.DataFrame({"atom": atom, \
                            "x": xyz[:, 0], \
                            "y": xyz[:, 1], \
                            "z": xyz[:, 2]})
            #df.to_csv("_input.xyz", header=None, index=None, sep=" "*6)
            try:
                mol_name = self.mol[ii].GetProp("_Name")
            except Exception as e:
                mol_name = "MOL"
            
            #with open(os.path.join(self.workdir, f"_input_{ii}.xyz"), "w+") as ff:
                #ff.write(f"{xyz.shape[0]}\n")
                #ff.write(f"{mol_name}\n")
            this_xyz = ""
            for idx, row in df.iterrows():
                this_xyz += f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n"
            
            self.xyz_blocks.append((this_xyz, int(charge), ii))
        
        #print([cc[1] for cc in self.xyz_blocks])
        

    
    def run_pyscf_each(self, xyz_item_set):
        _collect = []
        for xyz_item in xyz_item_set:
            mol = gto.Mole()
            mol.verbose = self.verbose
            mol.atom = xyz_item[0]
            mol.basis = self.basis
            mol.charge = xyz_item[1]
            mol.symmetry = False
            mol.spin = 0
            mol.build()

            if not self.if_D3:
                if not self.if_solvent:
                    mf = dft.RKS(mol, xc=self.functional)
                else:
                    mf = solvent.ddCOSMO(dft.RKS(mol,xc=self.functional)) 
            else:
                if not self.if_solvent:
                    mf = dftd3(dft.RKS(mol, xc=self.functional))
                else:
                    mf = solvent.ddCOSMO(dftd3(dft.RKS(mol,xc=self.functional)))

            try:
                e = mf.kernel()
            except Exception as e:
                e = math.inf
            _collect.append((e, xyz_item[-1]))

        return _collect
    
    def run_pyscf(self):
        if len(self.mol) <= self.cpu_core:
            n_jobs = len(self.mol)
            N_mol = 1
        else:
            n_jobs = self.cpu_core
            N_mol = math.ceil(len(self.mol)/self.cpu_core)
        
        collect = Parallel(n_jobs=n_jobs)(\
                delayed(self.run_pyscf_each)(self.xyz_blocks[ii*N_mol:(ii+1)*N_mol]) for ii in range(n_jobs))
        
        getFinal = collect[0]
        for ii in range(1, len(collect)):
            getFinal += collect[ii]
        getFinalSorted = sorted(getFinal, key=lambda x:x[0])

        #return (getFinalSorted[0], self.mol[getFinalSorted[0][-1]])
        #return collect
        os.system("rm -f tmp*")
        return getFinalSorted, self.mol
    
    def run(self, method):
        run_dict = {"pyscf": self.run_pyscf}

        try:
            run_dict[method]
        except Exception as e:
            logging.info(f"Wrong method setting, please choose from [{[kk for kk in run_dict.keys()]}]")
            return None
        
        if self.mol:
            final = run_dict[method]()
        
        #return final
