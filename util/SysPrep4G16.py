import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import logging
import math
import json

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class main():
    def __init__(self, **args):
        try:
            self.db_name = args["input_sdf"]
        except Exception as e:
            self.db_name = "_INPUT.sdf"
            try:
                self.mol = args["input_rdmol_obj"]
                if not isinstance(self.mol, list):
                    self.mol = [self.mol]
            except Exception as e:
                self.mol = None
        else:
            self.mol = [mm for mm in Chem.SDMolSupplier(self.db_name, removeHs=False) if mm]

        self.mode = args["mode"]

        self._default_functional = {"opt": "B3LYP", \
                                    "sp": "B2PLYPD3"}
        self._default_basis = {"opt": "6-311G*", \
                               "sp": "def2TZVP"}

        try:
            self.platform = args["platform"]
        except Exception as e:
            self.platform = ''

        try:
            self.functional_opt = args["functional_opt"]
        except Exception as e:
            self.functional_opt = self._default_functional["opt"]
        
        try:
            self.functional_sp = args["functional_sp"]
        except Exception as e:
            self.functional_sp = self._default_functional["sp"]
        
        try:
            self.basis_opt = args["basis_opt"]
        except Exception as e:
            self.basis_opt = self._default_basis["opt"]
        
        try:
            self.basis_sp = args["basis_sp"]
        except Exception as e:
            self.basis_sp = self._default_basis["sp"]
        
        try:
            self.solvation_model = args["solvation_model"]
        except Exception as e:
            self.solvation_model = ''

        try:
            self.if_d3 = args["if_d3"]
        except Exception as e:
            self.if_d3 = True
        
        try:
            self.if_chk = args["if_chk"]
        except Exception as e:
            self.if_chk = True
        
        try:
            self.nproc = int(args["nproc"])
        except Exception as e:
            self.nproc = 16
        else:
            if self.nproc <= 0:
                self.nproc = 16
        
        try:
            self.mem = int(args["mem"])
        except Exception as e:
            self.mem = 20
        else:
            if self.mem <= 0:
                self.mem = 20
        
        try:
            self.define_charge = args["define_charge"]
        except Exception as e:
            self.define_charge = None

        
        try:
            self.PJ_id = cc_agrs["project_id"]
        except Exception as e:
            self.PJ_id = ''
        
        self._default_machine_type = {16: "c16_m128_cpu", 
                                      32: "c32_m256_cpu"}
        
        try:
            self.machine_type = args["machine_type"]
        except Exception as e:
            if self.nproc not in [cc for cc in self._default_machine_type.keys()]:
                self.machine_type = 'c16_m128_cpu'
            else:
                self.machine_type = self._default_machine_type[self.nproc]

    
    def sanitize(self, rdmol_obj, prefix):
        try:
            re_rdmol_obj = Chem.RemoveHs(rdmol_obj)
        except Exception as e:
            pass
        
        atom = [atom.GetSymbol() for atom in rdmol_obj.GetAtoms()]
        re_atom = [atom.GetSymbol() for atom in re_rdmol_obj.GetAtoms()]

        if len(atom) == len(re_atom):
            if len([cc for cc in atom if "H" in cc]) > 1:
                return rdmol_obj
            else:
                cc = Chem.SDWriter(f"TEMP_{prefix}.sdf")
                cc.write(rdmol_obj)
                cc.close()
                os.system(f"obabel -isdf TEMP_{prefix}.sdf -O TEMP_{prefix}_H.sdf -h")
        else:
            return rdmol_obj
        
        if os.path.isfile(f"TEMP_{prefix}_H.sdf"):
            new_mol = [mmm for mmm in Chem.SDMolSupplier(f"TEMP_{prefix}_H.sdf", removeHs=False) if mmm][0]
            os.system("rm -f TEMP_*")
            return new_mol
        else:
            return None
    
    def get_prop(self, rdmol_obj):
        atom = [atom.GetSymbol() for atom in rdmol_obj.GetAtoms()]
        xyz = rdmol_obj.GetConformer().GetPositions()

        if self.define_charge:
            try:
                charge = int(self.define_charge)
            except Exception as e:
                pass
            else:
                return atom, xyz, int(charge)
        
        AllChem.ComputeGasteigerCharges(rdmol_obj)
        _charge = sum([float(atom.GetProp("_GasteigerCharge")) for atom in rdmol_obj.GetAtoms()])

        if _charge:
            charge_sign = _charge / abs(_charge)

            if math.ceil(abs(_charge)) - abs(_charge) < 5e-1:
                charge = math.ceil(abs(_charge)) * charge_sign
            else:
                charge = (math.ceil(abs(_charge)) - 1)* charge_sign
        else:
            charge = int(_charge)
        
        return atom, xyz, int(charge)
    
    def write_gjf_binary(self, rdmol_obj, prefix):
        true_mol = self.sanitize(rdmol_obj, prefix)
        atom, xyz, charge = self.get_prop(true_mol)

        df = pd.DataFrame({"atom": atom, \
                            "x": xyz[:, 0], \
                            "y": xyz[:, 1], \
                            "z": xyz[:, 2]})

        with open(f"{prefix}.gjf", "w+") as c:
            if self.if_chk:
                c.write(f"%chk={prefix}.chk\n")
            
            c.write(f"%mem={self.mem}GB\n")
            c.write(f"%nproc={self.nproc}\n")
        
            if self.if_d3:
                cmd = f"# {self.functional_opt}/{self.basis_opt} em=GD3BJ opt freq {self.solvation_model} nosymm \n"
            else:
                cmd = f"# {self.functional_opt}/{self.basis_opt} opt freq {self.solvation_model} nosymm \n"
            c.write(cmd)
            c.write("\n")
            c.write(f"opt {prefix}\n")
            c.write("\n")
            c.write(f"{charge} 1\n")

            for idx, row in df.iterrows():
                c.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")

            c.write("\n")
            ## link section
            c.write("--link1-- \n")
            c.write(f"%chk={prefix}.chk\n")
            c.write(f"%mem={self.mem}GB\n")
            c.write(f"%nproc={self.nproc}\n")
            c.write(f"# {self.functional_sp}/{self.basis_sp} scrf(SMD,solvent=water) nosymm geom=allcheck \n")
            c.write("\n")
            c.write("\n")
    
    def write_single_opt(self, rdmol_obj, prefix):
        true_mol = self.sanitize(rdmol_obj, prefix)
        atom, xyz, charge = self.get_prop(true_mol)

        df = pd.DataFrame({"atom": atom, \
                            "x": xyz[:, 0], \
                            "y": xyz[:, 1], \
                            "z": xyz[:, 2]})

        with open(f"{prefix}.gjf", "w+") as c:
            if self.if_chk:
                c.write(f"%chk={prefix}.chk\n")
            
            c.write(f"%mem={self.mem}GB\n")
            c.write(f"%nproc={self.nproc}\n")
        
            if self.if_d3:
                cmd = f"# {self.functional_opt}/{self.basis_opt} em=GD3BJ opt {self.solvation_model} nosymm \n"
            else:
                cmd = f"# {self.functional_opt}/{self.basis_opt} opt {self.solvation_model} nosymm \n"
            c.write(cmd)
            c.write("\n")
            c.write(f"opt {prefix}\n")
            c.write("\n")
            c.write(f"{charge} 1\n")

            for idx, row in df.iterrows():
                c.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")

            c.write("\n")
        return 

    def write_single_sp(self, rdmol_obj, prefix):
        true_mol = self.sanitize(rdmol_obj, prefix)
        atom, xyz, charge = self.get_prop(true_mol)

        df = pd.DataFrame({"atom": atom, \
                            "x": xyz[:, 0], \
                            "y": xyz[:, 1], \
                            "z": xyz[:, 2]})

        with open(f"{prefix}.gjf", "w+") as c:
            if self.if_chk:
                c.write(f"%chk={prefix}.chk\n")
            
            c.write(f"%mem={self.mem}GB\n")
            c.write(f"%nproc={self.nproc}\n")
        
            if self.if_d3:
                cmd = f"# {self.functional_opt}/{self.basis_opt} em=GD3BJ {self.solvation_model} nosymm \n"
            else:
                cmd = f"# {self.functional_opt}/{self.basis_opt} {self.solvation_model} nosymm \n"
            c.write(cmd)
            c.write("\n")
            c.write(f"sp {prefix}\n")
            c.write("\n")
            c.write(f"{charge} 1\n")

            for idx, row in df.iterrows():
                c.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")

            c.write("\n")
        return 
        
    def setup_local(self):
        with open("run.sh", "w+") as cc:
            cc.write("#!/bin/sh \n")
            cc.write("\n")
            cc.write("runfile=`ls *.gjf`\n")
            cc.write("for ff in ${runfile}\n")
            cc.write("do\n")
            cc.write("\tname=`basename $ff`\n")
            cc.write("\tg16 < $ff > ${name}.log\n")
            cc.write("done\n")
        logging.info("Prepared for local run, run by 'nohup bash run.sh &' to start running")
        return 

    def setup_lbg(self):
        lbg_json = {
            "job_name": "Flow_with_G16",
            "command": "bash run.sh",
            "log_file": "tmp_log",
            "backward_files": [],
            "program_id": self.PJ_id,
            "platform": "ali",
            "job_group_id": "",
            "disk_size": 128,
            "machine_type": self.machine_type,
            "job_type": "container",
            "image_name": "registry.dp.tech/dptech/prod-1364/strain:run0.0.2"
        }

        with open("input.json", "w+") as cc:
            json.dump(lbg_json, cc, indent=4)
        
        self.setup_local()
        
        return 

    def run(self, **args):
        #work_dir = os.getcwd()
        try:
            define_prefix = args["prefix"]
        except Exception as e:
            define_prefix = None

        _dic_platform = {"local": self.setup_local, \
                    "lbg": self.setup_lbg}

        _dic_run = {"binary": self.write_gjf_binary, \
                    "opt": self.write_single_opt, \
                    "sp": self.write_single_sp}

        if self.mode not in [kk for kk in _dic_run.keys()]:
            loggging.info(f"No available runing mode, should choose from {[kk for kk in _dic_run.keys()]}")

        for ii, mm in enumerate(self.mol):
            if define_prefix:
                this_prefix = f"{define_prefix}_{ii}"
            else:
                this_prefix = f"{self.db_name.split('.')[0]}_{ii}"
            _dic_run[self.mode](mm, this_prefix)
        
        if self.platform:
            if self.platform == "lbg":
                if not self.PJ_id:
                    logging.info("Please define project id in 'input.json' if you use lbg to run")
            _dic_platform[self.platform]()

        return


