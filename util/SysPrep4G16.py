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
        self.db_name = args["input_sdf"]

        self.mol = [mm for mm in Chem.SDMolSupplier(self.db_name, removeHs=False) if mm]

        self.mode = args["mode"]

        self._default_functional = {"opt": "B3LYP", \
                                    "sp": "B2PLYPD3"}
        self._default_basis = {"opt": "6-311G*", \
                               "sp": "def2TZVP"}

        self.platform = args["platform"]

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
            self.PJ_id = cc_agrs["project_id"]
        except Exception as e:
            self.PJ_id = ''
        
        try:
            self.machine_type = args["machine_type"]
        except Exception as e:
            self.machine_type = 'c16_m128_cpu'

    
    def sanitize(self, rdmol_obj, prefix):
        cc = Chem.SDWriter(f"TEMP_{prefix}.sdf")
        cc.write(rdmol_obj)
        cc.close()
        ## detect if sanitize
        re_rdmol_obj = Chem.RemoveHs(rdmol_obj)
        if Chem.MolToSmiles(re_rdmol_obj) == Chem.MolToSmiles(rdmol_obj):
            #need to add H
            os.system(f"obabel -isdf TEMP_{prefix}.sdf -O TEMP_{prefix}_H.sdf -h")
        else:
            ## no need to add H
            os.system(f"obabel -isdf TEMP_{prefix}.sdf -O TEMP_{prefix}_H.sdf")
        
        if os.path.isfile(f"TEMP_{prefix}_H.sdf"):
            new_mol = [mmm for mmm in Chem.SDMolSupplier(f"TEMP_{prefix}_H.sdf", removeHs=False) if mmm][0]
            os.system("rm -f TEMP_*")
            return new_mol
        else:
            loggging.info(f"Fail to get mol {prefix} in sdf database")
            os.system("rm -f TEMP_*")
            return None
    
    def get_prop(self, rdmol_obj):
        atom = [atom.GetSymbol() for atom in rdmol_obj.GetAtoms()]
        xyz = rdmol_obj.GetConformer().GetPositions()

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
    
    def write_single_opt(self):
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

    def write_single_sp(self):
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

    def run(self):
        #work_dir = os.getcwd()

        _dic_platform = {"local": self.setup_local, \
                    "lbg": self.setup_lbg}

        _dic_run = {"binary": self.write_gjf_binary, \
                    "opt": self.write_single_opt, \
                    "sp": self.write_single_sp}

        if self.mode not in [kk for kk in _dic_run.keys()]:
            loggging.info(f"No available runing mode, should choose from {[kk for kk in _dic_run.keys()]}")
        
        if self.platform == "lbg":
            if not self.PJ_id:
                logging.info("Please define project id in 'input.json' if you use lbg to run")

        for ii, mm in enumerate(self.mol):
            this_prefix = f"{self.db_name.split('.')[0]}_{ii}"
            _dic_run[self.mode](mm, this_prefix)
        
        _dic_platform[self.platform]()

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process g16 run')
    parser.add_argument('--input_sdf', type=str, required=True, help='input sdf file')
    parser.add_argument('--mode', type=str, required=True, help='run type, could be [sp, opt, binary]')
    parser.add_argument('--platform', type=str, required=True, help='platform of run, could be [local, lbg]')
    args = parser.parse_args()

    main(input_sdf=args.input_sdf, mode=args.mode, platform=args.platform).run()

