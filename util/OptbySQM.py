#!/usr/bin/env python
# coding: utf-8

"""
// author: Wang (Max) Jiayue 
// email: wangjy108@outlook.com
// git profile: https://github.com/wangjy108
"""

from rdkit import rdBase, Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdmolfiles
import pandas as pd
import numpy as np
import os
import logging
import math
import subprocess
from joblib import Parallel, delayed
from util.CalRMSD import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
// input sdf, transform xyz and perform xtb opt
// return rdkit mol set
"""

class System():
    def __init__(self, **args):
        try:
            self.mol = [mm for mm in Chem.SDMolSupplier(args["input_sdf"], removeHs=False) if mm]
        except Exception as e:
            logging.info("Wrong input, check and run again")
            self.mol = None
        
        self.db_prefix = args["input_sdf"].split(".")[0]

        self.workdir = os.getcwd()

        try:
            self.level = args["level"]
        except Exception as e:
            self.level = 'normal'
        else:
            if not self.level in ['crude', 'sloppy', 'loose', 'lax', 'tight', 'normal', 'vtight', 'extreme']:
                self.level = 'normal'
        
        try:
            self.gfn_option = int(args["gfn_option"])
        except Exception as e:
            self.gfn_option = 2
        else:
            if int(self.gfn_option) not in range(0, 3):
                self.gfn_option = 2
        
        try:
            self.solvation = args["solvation"]
        except Exception as e:
            self.solvation = 'h2o'
        
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
            self.rmsd_cutoff = args["rmsd_cutoff"]
        except Exception as e:
            self.rmsd_cutoff = 1.0
        else:
            try:
                self.rmsd_cutoff = float(self.rmsd_cutoff)
            except Exception as e:
                self.rmsd_cutoff = 1.0
            
        try:
            self.qm_opt = args["qm_opt"]
        except Exception as e:
            self.qm_opt = False
        else:
            if not isinstance(self.qm_opt, bool):
                self.qm_opt = False
        
        try:
            self.energy_gap = args["energy_gap"]
        except Exception as e:
            self.energy_gap = 5.0
        else:
            try:
                self.energy_gap = float(self.energy_gap)
            except Exception as e:
                self.energy_gap = 5.0
        
        self.command_line = []
        ## prepare input xyz, sample from obabel tranformation
        for ii in range(len(self.mol)):
            if not os.path.exists(os.path.join(self.workdir, f"run_{ii}")):
                os.mkdir(os.path.join(self.workdir, f"run_{ii}"))

            atom = [atom.GetSymbol() for atom in self.mol[ii].GetAtoms()]
            xyz = self.mol[ii].GetConformer().GetPositions()

            ## get charge
            AllChem.ComputeGasteigerCharges(self.mol[ii])
            _charge = sum([float(atom.GetProp("_GasteigerCharge")) for atom in self.mol[ii].GetAtoms()])

            charge_sign = _charge / abs(_charge)

            if math.ceil(abs(_charge)) - abs(_charge) < 5e-1:
                charge = math.ceil(abs(_charge)) * charge_sign
            else:
                charge = (math.ceil(abs(_charge)) - 1)* charge_sign
        
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
            
            with open(os.path.join(self.workdir, f"run_{ii}/_input_{ii}.xyz"), "w+") as ff:
                ff.write(f"{xyz.shape[0]}\n")
                ff.write(f"{mol_name}\n")
                for idx, row in df.iterrows():
                    ff.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")
            
            self.command_line.append(f"xtb _input_{ii}.xyz --opt --chrg {int(charge)} --gfn {self.gfn_option} --gbsa {self.solvation} > _log")
            
    def run_set(self, sub_set:list):
        collect = {}
        for each in sub_set:
            real_idx = int(each.split()[1].split(".")[0].split("_")[-1])
            path = os.path.join(self.workdir, f"run_{real_idx}")
            os.chdir(path)
            (status, output) = subprocess.getstatusoutput(each)
            if status == 0 and 'normal termination of xtb' in output:
                ## normal termination, collect result
                mol_opt = rdmolfiles.MolFromXYZFile(os.path.join(path,"xtbopt.xyz"))
                ## collect energy
                with open(os.path.join(path,"xtbopt.xyz"), "r+") as ff:
                    conetent = [line.strip() for line in ff.readlines()] 
                    getEnergy = float(conetent[1].split()[1])
                #logging.info(str(getEnergy))

                #mol_opt.SetProp("Energy_xtb", f"{getEnergy}")
                #mol_opt.SetProp("_Name", f"{self.db_prefix}_{real_idx}")
                collect.setdefault(real_idx, (mol_opt, getEnergy, f"{self.db_prefix}_{real_idx}"))
            else:
                collect.setdefault(real_idx, None)
            os.chdir(self.workdir)
        
        return collect
    
    def run_opt(self):
        #os.syetm(f"export OMP_NUM_THREADS=4")
        batch = math.ceil(len(self.command_line) / 4)
        if batch <= 4:
            OMP = self.cpu_core // batch
        else:
            OMP = 4
        os.system(f"export OMP_NUM_THREADS={OMP}")
        assemble = Parallel(n_jobs=batch)(\
                delayed(self.run_set)(self.command_line[ii*4:(ii+1)*4]) for ii in range(batch))
        
        os.system("rm -rf run_*")
        dict_assemble = assemble[0]
        for ii in range(1,len(assemble)):
            dict_assemble.update(assemble[ii])
        
        return dict_assemble
    
    def run_process(self):
        dict_assemble = self.run_opt()
        if [kk for kk in dict_assemble.keys() if not dict_assemble[kk]]:
            logging.info(f"xtb opt failed with {[kk for kk in dict_assemble.keys() if not dict_assemble[kk]]}th mol in input sdf db")
        
        standar_save = []

        ## sort optimized mol according to energy
        optimized = sorted([vv for vv in dict_assemble.values() if vv], key=lambda x:x[1])
        if not self.qm_opt:
            ## standar sdf
            save = optimized[:1]
            
        else:
            energy_range = [x[1] for x in optimized]
            energy_range_standarized = [(cc - min(energy_range))*627.51 for cc in energy_range]
            if max(energy_range_standarized) < self.energy_gap:
                for_align = optimized
            else:
                ii = 1
                for_align = optimized[:1]
                while ii < len(optimized):
                    if energy_range_standarized[ii] >= self.energy_gap:
                        break
                    else:
                        ii += 1
                for_align += optimized[1:ii]
            
            ## do align and remove those with rmsd less than self.rmsd_cutoff
            
            save = for_align[:1]
            for ii in range(len(for_align)):
                compare_rmsd = [RMSD(rdmolobj_mol=for_align[ii][0],rdmolobj_ref=save[cc][0], method="selfWhole").run() for cc in range(len(save))]
                if min(compare_rmsd) > self.rmsd_cutoff:
                    save.append(for_align[ii])
            
        
        for each in save:
            real_mol = each[0]
            real_mol.SetProp("Energy_xtb", str(each[1]))
            real_mol.SetProp("_Name", each[-1])
            standar_save.append(real_mol)
        
        return standar_save
                

    
                


        
    
    


        
