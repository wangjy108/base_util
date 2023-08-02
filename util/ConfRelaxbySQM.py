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


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
// input sdf single, transform xyz and perform MD
"""

class System():
    def __init__(self, **args):
        try:
            self.mol = [mm for mm in Chem.SDMolSupplier(args["input_sdf"], removeHs=False) if mm][0]
        except Exception as e:
            logging.info("Wrong input, check and run again")
            self.mol = None
        
        with open(args["input_sdf"], "r+") as f1:
            self.ori_content = [ff for ff in f1.readlines()]

        try:
            self.run_time = args["run_time"]
        except Exception as e:
            self.run_time = 50
        else:
            try:
                self.run_time = float(self.run_time)
            except Exception as e:
                self.run_time = 50
        
        try:
            self.run_temperature = args["run_temperature"]
        except Exception as e:
            self.run_temperature = 400
        else:
            try:
                self.run_temperature = int(self.run_temperature)
            except Exception as e:
                self.run_temperature = 400
        
        try:
            self.charge = args["define_charge"]
        except Exception as e:
            self.charge = None

        
        try:
            self.n_mpi = args["n_mpi"]
        except Exception as e:
            self.n_mpi = 16
        else:
            try:
                self.n_mpi = int(self.n_mpi)
            except Exception as e:
                self.n_mpi = 16
        
        ## default saving frame should be less than 1000
        #self.dump = int(self.run_time)
        try:
            self.save_frame = int(args["save_frame"])
        except Exception as e:
            self.save_frame = 100

        self.dump = int(self.run_time * 1000 / self.save_frame)
        
        ## prepare input xyz, sample from obabel tranformation
        self.atom = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.xyz = self.mol.GetConformer().GetPositions()

        ## get charge
        AllChem.ComputeGasteigerCharges(self.mol)
        if not self.charge:
            _charge = sum([float(atom.GetProp("_GasteigerCharge")) for atom in self.mol.GetAtoms()])

            if _charge:
                charge_sign = _charge / abs(_charge)

                if math.ceil(abs(_charge)) - abs(_charge) < 5e-1:
                    charge = math.ceil(abs(_charge)) * charge_sign
                else:
                    charge = (math.ceil(abs(_charge)) - 1)* charge_sign
                self.charge = int(charge)
                
            else:
                self.charge = int(_charge)
        else:
            self.charge = int(self.charge)
        
        ## save input.xyz for xtb-MD run

        df = pd.DataFrame({"atom": self.atom, \
                           "x": self.xyz[:, 0], \
                           "y": self.xyz[:, 1], \
                           "z": self.xyz[:, 2]})
        #df.to_csv("_input.xyz", header=None, index=None, sep=" "*6)
        try:
            self.mol_name = self.mol.GetProp("_Name")
        except Exception as e:
            self.mol_name = "MOL"
        
        with open("_input.xyz", "w+") as ff:
            ff.write(f"{self.xyz.shape[0]}\n")
            ff.write(f"{self.mol_name}\n")
            for idx, row in df.iterrows():
                ff.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")

        with open("md.inp", "w+") as cc:
            cc.write("$md \n")
            cc.write(f"temp = {self.run_temperature} \n")
            cc.write(f"time = {self.run_time:.1f}\n")
            cc.write(f"dump = {self.dump:.1f} \n")
            cc.write("step = 1.0 \n")
            cc.write("shake = 1 \n")
            cc.write("hmass = 1 \n")
            cc.write("$end \n")

    def shift_sdf(self, update_sdf):
        ## ori content: self.ori_content,
        ## ori mol: self.mol

        ori_content = self.ori_content
        rdmolobj_ori = self.mol

        with open(update_sdf, "r+") as f2:
            upt_content = [ff for ff in f2.readlines()]
        
        rdmolobj_upt = [mm for mm in Chem.SDMolSupplier(update_sdf, removeHs=False) if mm][0]

        ## header should stop at 
        xyz_upt = rdmolobj_upt.GetConformer().GetPositions()
        xyz_ori = rdmolobj_ori.GetConformer().GetPositions()
        ## xyz_upt[-1][0]
        
        upper_end_idx = [ii for ii in range(len(upt_content)) if upt_content[ii].strip().endswith("END")][-1]
        
        middle_start_idx = [ii for ii in range(len(ori_content)) if ori_content[ii].strip().startswith(str(xyz_ori[-1][0]))][-1] + 1
        middle_end_idx = [ii for ii in range(len(ori_content)) if ori_content[ii].strip().endswith("END")][-1]

        header_replace_idx_upt = [ii for ii in range(len(upt_content)) \
                                if upt_content[ii].strip().startswith(str(xyz_upt[0][0]))][0] -1
        header_replace_idx_ori = [ii for ii in range(len(ori_content)) \
                                if ori_content[ii].strip().startswith(str(xyz_ori[0][0]))][0] -1
        
        upper = upt_content[:upper_end_idx]
        upper[header_replace_idx_upt] = ori_content[header_replace_idx_ori]

        assemble_content = upper \
                        + ori_content[middle_start_idx:middle_end_idx] \
                        + upt_content[upper_end_idx:]
        
        return assemble_content

    def process_traj(self, traj):
        save = []
        _track = 0
        with open(traj, "r+") as ff:
            content = [line for line in ff.readlines() if line.strip()]
        
        atom_number = self.xyz.shape[0]

        geom_idx = [ii for ii in range(len(content)) if content[ii].strip() == str(atom_number)]

        for idx, geom in enumerate(geom_idx):
            cc = Chem.SDWriter(f"_TEMP_{idx}.sdf")
            try:
                this_xyz = content[geom_idx[idx]:geom_idx[idx+1]]
            except Exception as e:
                this_xyz = content[geom_idx[idx]:]
            
            xyz_block = ""
            for line in this_xyz:
                xyz_block += line
            
            this_mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            this_energy = content[geom_idx[idx]+1].split()[1]
            this_mol.SetProp("_Name", f"MDrelax_{idx}")
            this_mol.SetProp("Energy_gfnxtb", this_energy)
            cc.write(this_mol)
            cc.close()

            try:
                get_content = self.shift_sdf(f"_TEMP_{idx}.sdf")
            except Exception as e:
                continue
            save += get_content
            _track += 1
        
        with open("SAVE.sdf", "w+") as cc:
            for each in save:
                cc.write(each)

        logging.info("Save all dumped geomtries in SAVE.sdf")
        os.system("rm -f sc* gfn* .xtb* xtbmdok *.log *.xyz md* _TEMP* _log")

        if _track == len(geom_idx):
            os.system(f"rm -f {traj}")
        else:
            logging.info(f"Not properly save, check '{traj}' for all geom information")
        
        return 

    def run(self):
        os.system(f"export OMP_NUM_THREADS={self.n_mpi}")
        command = f"xtb _input.xyz --input md.inp --chrg {self.charge} --omd --gfnff > _log"
        logging.info(f"run command: {command}")
        (status, output) = subprocess.getstatusoutput(command)

        if status == 0 and 'normal termination of xtb' in output:
            self.process_traj("xtb.trj")  
        else:
            logging.info("Failed at xtb md for conformer sampling, nothing would be save")          
