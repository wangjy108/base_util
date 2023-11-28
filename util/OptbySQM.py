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
from util.Cluster import cluster

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
// input sdf, transform xyz and perform xtb opt
// return rdkit mol set and saved sdf file
"""

class System():
    def __init__(self, **args):
        try:
            self.mol = [mm for mm in Chem.SDMolSupplier(args["input_sdf"], removeHs=False) if mm]
        except Exception as e:
            #logging.info("Wrong input, check and run again")
            self.mol = None
        
        if not self.mol:
            try:
                self.mol = args["input_rdmol_obj"]
            except Exception as e:
                self.mol = None
        
        #self.db_prefix = args["input_sdf"].split(".")[0]
        self.db_prefix = "OPT"

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
            self.save_n = args["save_n"]
        except Exception as e:
            self.save_n = None

        """    
        try:
            self.qm_opt = args["qm_opt"]
        except Exception as e:
            self.qm_opt = False
        else:
            if not isinstance(self.qm_opt, bool):
                self.qm_opt = False
        """
        
        try:
            self.HA_constrain = args["HA_constrain"]
        except Exception as e:
            self.HA_constrain = False
        else:
            if not isinstance(self.HA_constrain, bool):
                self.HA_constrain = False
        
        try:
            charge = args["define_charge"]
        except Exception as e:
            charge = None
        
        try:
            self.if_write_sdf = args["if_write_sdf"]
        except Exception as e:
            self.if_write_sdf = False
        
        self.command_line = []
        ## prepare input xyz, sample from obabel tranformation
        for ii in range(len(self.mol)):
            if not os.path.exists(os.path.join(self.workdir, f"run_{ii}")):
                os.mkdir(os.path.join(self.workdir, f"run_{ii}"))

            atom = [atom.GetSymbol() for atom in self.mol[ii].GetAtoms()]
            xyz = self.mol[ii].GetConformer().GetPositions()

            ## get charge
            AllChem.ComputeGasteigerCharges(self.mol[ii])
            if not charge:
                _charge = sum([float(atom.GetProp("_GasteigerCharge")) for atom in self.mol[ii].GetAtoms()])

                if _charge:
                    charge_sign = _charge / abs(_charge)

                    if math.ceil(abs(_charge)) - abs(_charge) < 5e-1:
                        charge = math.ceil(abs(_charge)) * charge_sign
                    else:
                        charge = (math.ceil(abs(_charge)) - 1)* charge_sign
                
                else:
                    charge = int(_charge)
            else:
                charge = int(charge)
        
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
            
            if self.HA_constrain:
                HA_atom_type = set(atom)
                atom_line = "elements: "
                for each in HA_atom_type:
                    if "H" not in each:
                        atom_line += f"{each},"

                atom_line = atom_line[:-1]

                with open(os.path.join(self.workdir, f"run_{ii}/_input_{ii}.inp"), "w+") as fff:
                    fff.write("$constrain\n")
                    fff.write(f"{atom_line}\n")
                    fff.write(f"$end\n")

                self.command_line.append(f"xtb _input_{ii}.xyz --input _input_{ii}.inp --opt \
                                          --chrg {int(charge)} --gfn {self.gfn_option} --gbsa {self.solvation} --verbose > _log")
            
            else:
                self.command_line.append(f"xtb _input_{ii}.xyz --opt \
                                           --chrg {int(charge)} --gfn {self.gfn_option} --gbsa {self.solvation} --verbose > _log")
            
    def run_set(self, sub_set:list):
        collect = {}
        for each in sub_set:
            real_idx = int(each.split()[1].split(".")[0].split("_")[-1])
            get_charge = int(each.split()[-8])
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
                collect.setdefault(real_idx, (mol_opt, getEnergy, f"{self.db_prefix}_{real_idx}", get_charge))
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
    
    def shift_sdf(self, original_sdf, update_sdf, save_prefix):
        with open(original_sdf, "r+") as f1:
            ori_content = [ff for ff in f1.readlines()]

        rdmolobj_ori = [mm for mm in Chem.SDMolSupplier(original_sdf, removeHs=False) if mm][0]
        
        with open(update_sdf, "r+") as f2:
            upt_content = [ff for ff in f2.readlines()]
        
        rdmolobj_upt = [mm for mm in Chem.SDMolSupplier(update_sdf, removeHs=False) if mm][0]

        ## header should stop at 
        xyz_upt = rdmolobj_upt.GetConformer().GetPositions()
        xyz_ori = rdmolobj_ori.GetConformer().GetPositions()
        ## xyz_upt[-1][0]
        
        ## direct get -1 may also cost invisible problems as one may include comments in the following contens
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
        
        with open(f"{save_prefix}.sdf", "w+") as cc:
            for line in assemble_content:
                cc.write(line)
        
        return 
    
    def run_process(self):
        dict_assemble = self.run_opt()
        if [kk for kk in dict_assemble.keys() if not dict_assemble[kk]]:
            logging.info(f"xtb opt failed with {[kk for kk in dict_assemble.keys() if not dict_assemble[kk]]}th mol in input sdf db")
        
        standard_save = []
        standard_charge = []

        ## do cluster
        before_cluster = []
        for kk, vv in dict_assemble.items():
            if vv:
                vv[0].SetProp("Energy_xtb", str(vv[1]))
                vv[0].SetProp("_Name", vv[2])
                vv[0].SetProp("charge", str(vv[3]))
                before_cluster.append(vv[0])
        
        if len(before_cluster) > 1:
            ## reduce_duplicate
            after_cluster = cluster(input_rdmol_obj=before_cluster,
                                    rmsd_cutoff_cluster=self.rmsd_cutoff,
                                    do_align=True,
                                    only_reduce_duplicate=True).run()
        else:
            after_cluster = before_cluster

        optimized = sorted([mm for mm in after_cluster], key=lambda x:float(x.GetProp("Energy_xtb")))

        if not self.save_n:
            return optimized
        
        return optimized[:self.save_n]
    
    def run(self):
        out_put = []
        optmized = self.run_process()

        for each in optmized:
            real_idx = int(each.GetProp("_Name").split("_")[-1])

            save_ori = Chem.SDWriter(f"_TEMP_ori_{real_idx}.sdf")
            save_ori.write(self.mol[real_idx])
            save_ori.close()
            save_upt = Chem.SDWriter(f"_TEMP_upt_{real_idx}.sdf") 
            save_upt.write(each)
            save_upt.close()

            self.shift_sdf(original_sdf=f"_TEMP_ori_{real_idx}.sdf", \
                           update_sdf=f"_TEMP_upt_{real_idx}.sdf", \
                           save_prefix=f"_TEMP_opt_{real_idx}")

            try:
                standard_real_mol = [mm for mm in Chem.SDMolSupplier(f"_TEMP_opt_{real_idx}.sdf", removeHs=False) if mm][0]
            except Exception as e:
                out_put.append(each)
            else:
                out_put.append(standard_real_mol)
            
        if self.if_write_sdf:
            final_cc = Chem.SDWriter("_OPT.sdf")
            for mol in out_put:
                final_cc.write(mol)
            final_cc.close()
            logging.info(f"Optimized mol saved in {os.getcwd()}/_OPT.sdf")

        os.system("rm -f _TEMP*")

        return out_put

                


        
    
    


        
