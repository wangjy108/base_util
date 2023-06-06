import pandas as pd
import os
import random
import math
import numpy as np
import time
import argparse
import sys
import configparser
import logging
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdMolTransforms

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from decimal import Decimal
from decimal import ROUND_HALF_UP,ROUND_HALF_EVEN
import subprocess
from collections import Counter

from util.SysPrep4G16 import main as G16

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class main():
    def __init__(self, **args):
        param_config = args["config"]
        self._input = args["input_sdf"] ## initial sdf
        #self.prefix = _input.split(".")[0]

        config = configparser.ConfigParser()
        config.read(param_config)
        general = config["general"]
        
        self.qm_opt_engine = general["QM_OPT_ENGINE"]
        self.qm_scan_engine = general["QM_SCAN_ENGINE"]
        self.charge = general["CHARGE"]
        self.spin = int(general["SPIN"])
        self.bond1 = str(general["BOND_1"])
        self.rotable_angle1 = int(general["ROTATE_ANGlE_1"])
        self.rotable_terval1 = int(general["TERVAL_1"])
        try:
            self.bond2 = str(general["BOND_2"])
        except Exception as e:
            self.bond2 = None
        else:
            if self.bond2:
                try:
                    self.rotable_angle2 = int(args["ROTATE_ANGlE_2"])
                except Exception as e:
                    self.rotable_angle2 = self.rotable_angle1
                
                try:
                    self.rotable_terval2 = int(args["TERVAL_2"])
                except Exception as e:
                    self.rotable_terval2 = self.rotable_terval1
                
            else:
                self.bond2 = None
 
        self.functional = general["FUNCTIONAL"]
        
        self.if_d3 = bool(int(general["D3"]))
        
        self.opt_basis = general["OPT_BASIS"]
        self.scan_basis = general["SCAN_BASIS"]
        #self.run_base = general["RUN_BASE"]
        
        self.if_chk = bool(int(general["CHK"]))

        run_config = config["run_config"]

        try:
            self.nproc = int(run_config["NPROC"])
        except Exception as e:
            self.nproc = 32
        
    
    def run_initial(self):
        ## initial opt and docking sp
        try:
            input_mol = [cc for cc in Chem.SDMolSupplier(self._input, removeHs=False) if cc][0]
        except Exception as e:
            cmd = f"obabel -isdf {self._input} -O _TMEP_input.xyz"
            (_status, _out) = subprocess.getstatusoutput(cmd)
            if not (_status == 0 and os.path.isfile("_TMEP_input.xyz") and os.path.getsize("_TMEP_input.xyz")):
                logging.info("Terrible input file, check and run again")
                return None
            try:
                input_mol = rdmolfiles.MolFromXYZFile("_TMEP_input.xyz")
            except Exception as e:
                logging.info("Terrible input file, check and run again")
                return None
        
        if input_mol:
            G16(input_rdmol_obj=input_mol, 
                        mode="opt", 
                        functional_opt=self.functional,
                        basis_opt=self.opt_basis, 
                        if_d3=self.if_d3, 
                        if_chk=self.if_chk, 
                        define_charge=self.charge, 
                        nproc=self.nproc).run(prefix="opt")
            G16(input_rdmol_obj=input_mol, 
                        mode="sp", 
                        functional_sp=self.functional,
                        basis_sp=self.scan_basis, 
                        if_d3=self.if_d3, 
                        if_chk=self.if_chk, 
                        define_charge=self.charge, 
                        nproc=self.nproc).run(prefix="sp")
        else:
            logging.info("Failed at generating initial input, check and run again")
            return None

        run_file = [ff for ff in os.listdir() if "opt" in ff and ".gjf" in ff]
        run_done_log = []
        if run_file:
            for _job in run_file:
                _out = str(_job).split(".")[0]
                cmd = f"g16 < {_job} > {_out}.log"
                (status, output) = subprocess.getstatusoutput(cmd)
                if status == 0 and os.path.isfile(f"{_out}.log"):
                    with open(f"{_out}.log", "r+") as ff:
                        tag = [hh for hh in ff.readlines() if hh.strip() and "Normal termination" in hh]
                    if tag:
                        logging.info(f"Optimization done for {_out}")
                        run_done_log.append(f"{_out}.log")
                    else:
                        logging.info(f"Optimization failed, check {_out}.log for more information")    
                else:
                    logging.info(f"Optimization failed, check {_out}.log for more information")
        else:
            logging.info("No available run file")
        
        return run_done_log

    def run_rotation(self, opt_run_done_log):
        ## do gentor scan
        ## get xyz block from log
        opt_log = [ll for ll in opt_run_done_log][0]
        _out = opt_log.split(".")[0]
        cmd = f"obabel -ig16 {opt_log} -O xyz"
        try:
            (status, output) = subprocess.getstatusoutput(cmd)
        except Exception as e:
            logging.info("Failed at get xyz block from g16 run_done file")
            return 0
        if status == 0:
            if os.path.isfile("xyz") and os.path.getsize("xyz"):
                ## run gentor to get trj file
                with open("gentor.ini", "w+") as cc:
                    simplified_bond1="-".join(self.bond1.strip().split(",")[1:3])
                    cc.write(simplified_bond1 + "\n")
                    if self.rotable_angle1 != 360:
                        write_terval1 = f"e{self.rotable_terval1} 0 {self.rotable_angle1}\n"
                    else:
                        write_terval1 = f"e{self.rotable_terval1}\n"
                    cc.write(write_terval1)
                    if self.bond2:
                        simplified_bond2="-".join(self.bond2.strip().split(",")[1:3])
                        if simplified_bond2 != simplified_bond1:
                            logging.info("Double scan mode")
                            cc.write(simplified_bond2 + "\n")

                            if self.rotable_angle2 != 360:
                                write_terval2 = f"e{self.rotable_terval2} 0 {self.rotable_angle2}\n"
                            else:
                                write_terval2 = f"e{self.rotable_terval2}\n"
                                
                            cc.write(write_terval2 + "\n")

        scan_gen_cmd = f"run_gentor"
        try:
            (_status, _output) = subprocess.getstatusoutput(scan_gen_cmd)
        except Exception as e:
            logging.info("Failed at geometry rotation")
            return 0
        else:
            if not (_status == 0 and os.path.isfile("traj.xyz")):
                logging.info("Failed at geometry rotation")
                return 0
            transform_cmd = "obabel -ixyz traj.xyz -O traj.sdf"
            try:
                (status, output) = subprocess.getstatusoutput(transform_cmd)
            except Exception as e:
                mol_set = self.transform_alternative("traj.xyz")
            else:
                if status == 0 and os.path.getsize("traj.sdf"):
                    try:
                        mol_set = [mm for mm in Chem.SDMolSupplier("traj.sdf", removeHs=False) if mm]
                    except Exception as e:
                        mol_set = self.transform_alternative("traj.xyz")
                    else:
                        if not mol_set:
                            mol_set = self.transform_alternative("traj.xyz")

                else:
                    mol_set = self.transform_alternative("traj.xyz")
        
        ## generate sp gjf file
        get_this_sys_prefix = "RigScan"
        G16(input_rdmol_obj=mol_set, 
            mode="sp", 
            functional_sp=self.functional, 
            basis_sp=self.scan_basis, 
            if_d3 = self.if_d3,
            if_chk = self.if_chk, 
            define_charge=self.charge, 
            nproc=self.nproc).run(prefix=get_this_sys_prefix)
        
        run_file = [ff for ff in os.listdir() if (get_this_sys_prefix  or "sp" in ff) and ".gjf" in ff]
        sp_run_done_log = []
        if run_file:
            for _job in run_file:
                _out = str(_job).split(".")[0]
                cmd = f"g16 < {_job} > {_out}.log"
                (status, output) = subprocess.getstatusoutput(cmd)
                if status == 0 and os.path.isfile(f"{_out}.log"):
                    with open(f"{_out}.log", "r+") as ff:
                        tag = [hh for hh in ff.readlines() if hh.strip() and "Job cpu time" in hh]
                    if tag:
                        #logging.info("Initial optimization done")
                        sp_run_done_log.append(f"{_out}.log")
                    else:
                        continue
                        #logging.info(f"Initial optimization failed, check {_out}.log for more information")
                else:
                    #logging.info(f"Initial optimization failed, check {_out}.log for more information")
                    continue
        else:
            logging.info("No available run file for scan, terminated")
            return 0
        
        return sp_run_done_log   

    def gen_conf_Multiwfn(self, param_list, input_file):
        tail = input_file.split(".")[0]
        with open(f"_Param_GF.{tail}", "w+") as cc:
            for line in param_list:
                cc.write(line + "\n")

        cmd = f"Multiwfn {input_file} < _Param_GF.{tail}"   
        (_status, _) = subprocess.getstatusoutput(cmd)
        return _status

    def transform_alternative(self, xyz_traj):
        save = []
        _track = 0
        with open(xyz_traj, "r+") as ff:
            content = [line for line in ff.readlines() if line.strip()]
        
        atom_number = int(content[0].strip())

        geom_idx = [ii for ii in range(len(content)) if content[ii].strip() == str(atom_number)]

        for idx, geom in enumerate(geom_idx):
            try:
                this_xyz = content[geom_idx[idx]:geom_idx[idx+1]]
            except Exception as e:
                this_xyz = content[geom_idx[idx]:]
            
            xyz_block = ""
            for line in this_xyz:
                xyz_block += line
            
            this_mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            save.append(this_mol)
            _track += 1
        
        if _track == len(geom_idx):
            os.system(f"rm -f {xyz_traj}")
            logging.info("Processing rotational points done")
            return save
        else:
            logging.info(f"Not properly save, check '{xyz_traj}' for all geom information")
            return 0

    def result_1D_scan(self, scan_done_log):
        ## should have some kind of _Param.log2dih
        _log_dih = {}
        self.dih_param()
        for log in scan_done_log:
            if "sp" not in log and "opt" not in log:
                _prefix = log.split(".")[0]
                with open(log, "r+") as ff:
                    content = [hh for hh in ff.readlines() if hh.strip()]
                
                validate_flag = [cc for cc in content if "Normal termination" in cc]
                
                if validate_flag:
                    ene = float([cc.strip() for cc in content if "SCF Done" in cc][0].split()[-5])
                    get_dih_cmd = f"Multiwfn {log} < _Param.log2dih.1 | grep 'The dihedral angle is' > _TMEP_{_prefix}.dih"
                    (_status, _) = subprocess.getstatusoutput(get_dih_cmd)
                    if _status == 0 and os.path.isfile(f"_TMEP_{_prefix}.dih") and os.path.getsize(f"_TMEP_{_prefix}.dih"):
                        with open(f"_TMEP_{_prefix}.dih", "r") as f:
                            line = [ff.strip() for ff in f.readlines()][0]
                            dih = Decimal(line.split()[-2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                            _log_dih.setdefault(log, (dih, ene))
        
         ## get docking pose sp energy
        _dock = {}
        try:
            dockP_log = [cc for cc in scan_done_log if "sp" in cc][0]
        except Exception as e:
            logging.info("Something went wrong for docking pose scan, check if sp*.gjf exist")
        else:
            with open(dockP_log, "r+") as ff:
                content = [hh for hh in ff.readlines() if hh.strip()]
            
            validate_flag = [cc for cc in content if "Normal termination" in cc]

            if validate_flag:
                ene = float([cc.strip() for cc in content if "SCF Done" in cc][0].split()[-5])
                get_dih_cmd = f"Multiwfn {dockP_log} < _Param.log2dih.1 | grep 'The dihedral angle is' > _TMEP_dockP.dih"
                (_status, _) = subprocess.getstatusoutput(get_dih_cmd)
                if _status == 0 and os.path.isfile("_TMEP_dockP.dih") and os.path.getsize("_TMEP_dockP.dih"):
                    with open("_TMEP_dockP.dih", "r+") as f:
                        line = [ff.strip() for ff in f.readlines()][0]
                        dih = Decimal(line.split()[-2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                        _dock.setdefault("Docking Pose", (dih, ene))
                
        ## there you have _log_dih
        if _log_dih:
            _log_dih_sort = sorted(_log_dih.items(), key=lambda x: x[-1][0])
            _ene_min = min([cc[-1] for cc in _log_dih.values()])
            
            _barrier_list = [(cc[-1][-1] - _ene_min)*627.51 for cc in _log_dih_sort]
            barrier = max(_barrier_list) - min(_barrier_list)

            X_torsion_angle = [cc[-1][0] for cc in _log_dih_sort]
            X = [i for i in range(len(X_torsion_angle))]

            X_smooth = np.linspace(max(X), min(X), 200)
            Y_smooth = make_interp_spline(X,_barrier_list)(X_smooth)

            plt.figure(figsize=(12,8), dpi=300)
            plt.plot(X, _barrier_list, 'bo')
            plt.plot(X_smooth, Y_smooth, 'g')

            plt.xticks(X, X_torsion_angle, rotation=30)

            plt.xlabel(f"{self.bond1} Rotation Angle/degree")
            plt.ylabel("Energy/kcal.mol-1")

            plt.savefig(f"Graph_RigidScan_1D.png")

            ## process docking pose
            if _dock:
                dockP_ene = (_dock["Docking Pose"][-1] - _ene_min) * 627.51
                save_barrier =[dockP_ene] + _barrier_list
                save_torsion_angle = [_dock["Docking Pose"][0]] + X_torsion_angle
                save_indicator = ["Docking Pose"] + [cc[0].split(".")[0] for cc in _log_dih_sort]
            else:
                save_indicator = [cc[0].split(".")[0] for cc in _log_dih_sort]
                save_torsion_angle = X_torsion_angle
                save_barrier = _barrier_list

            df = pd.DataFrame({"Pose/File Name": save_indicator, 
                               "Rotation Angle/degree":save_torsion_angle, 
                               "Energy/kcal.mol-1": save_barrier})
            df.to_csv(f"Data_RigidScan_1D.csv", index=None)
        else:
            logging.info("Failed at get final result, check *.log file to see if no convergence meet")
            return 

        ## seach minima and maxima
        minima = []
        maxima = []

        i = 0
        while i < len(_log_dih_sort):
            current_scope = _log_dih_sort[i-1:i+2]
            if len(current_scope) < 3:
                current_scope = _log_dih_sort[-1:] + _log_dih_sort[:i+2]
                if len(current_scope) < 3:
                    current_scope = _log_dih_sort[i-1:] + _log_dih_sort[:1]
                    
            pair = [cc[-1][-1] for cc in current_scope]
            
            if max([vv for vv in dict(Counter(pair)).values()]) > 1:
                pass
            else:
                if pair[1] == max(pair):
                    maxima.append(current_scope[1])
                elif pair[1] == min(pair):
                    minima.append(current_scope[1])
            i += 1
        
        sort_minima = sorted(minima, key=lambda x: x[-1][-1])
        sort_maxima = sorted(maxima, key=lambda x: x[-1][-1])

        _track_words = ""

        _track_words += "get global minima "

        global_minima_log = sort_minima[0][0]
        g_min_pdb_file_name = f"Conf_GlobalMinima_Angle:{sort_minima[0][-1][0]}.pdb"

        conf_gen_param = ["100", "2", "1", g_min_pdb_file_name, "0", "q"]
        _status = self.gen_conf_Multiwfn(conf_gen_param, global_minima_log)

        if _status == 0 and os.path.isfile(g_min_pdb_file_name) and os.path.getsize(g_min_pdb_file_name):
            _track_words += "done; "
        else:
            _track_words += "failed "
        
        if sort_minima[1:]:
            _track_words += "get local minima "
            _track_Lminima = []
            for each in sort_minima[1:]:
                local_minima_log = each[0]
                l_min_pdb_file_name = f"Conf_LocalMinima_Angle:{each[-1][0]}.pdb"
                conf_gen_param = ["100", "2", "1", l_min_pdb_file_name, "0", "q"]
                _status = self.gen_conf_Multiwfn(conf_gen_param, local_minima_log)
                if _status == 0 and os.path.isfile(l_min_pdb_file_name) and os.path.getsize(l_min_pdb_file_name):
                    _track_Lminima.append(local_minima_log)
            if len(_track_Lminima) == len(sort_minima[1:]):
                _track_words += "done; "
            else:
                _track_words += "except for "
                for cc in [vv[0] for vv in sort_minima[1:]]:
                    if cc not in _track_Lminima:
                        _track_words += f"{cc} "
            
        _track_words += "get global maxima "

        global_maxima_log = sort_maxima[-1][0]
        g_max_pdb_file_name = f"Conf_GlobalMaxima_Angle:{sort_maxima[-1][-1][0]}.pdb"

        conf_gen_param = ["100", "2", "1", g_max_pdb_file_name, "0", "q"]
        _status = self.gen_conf_Multiwfn(conf_gen_param, global_maxima_log)

        if _status == 0 and os.path.isfile(g_max_pdb_file_name) and os.path.getsize(g_max_pdb_file_name):
            _track_words += "done; "
        else:
            _track_words += "failed "
        
        if sort_maxima[:-1]:
            _track_words += "get local maxima "
            _track_Lmaxima = []
            for each in sort_maxima[:-1]:
                local_maxima_log = each[0]
                l_max_pdb_file_name = f"Conf_LocalMaxima_Angle:{each[-1][0]}.pdb"
                conf_gen_param = ["100", "2", "1", l_max_pdb_file_name, "0", "q"]
                _status = self.gen_conf_Multiwfn(conf_gen_param, local_maxima_log)
                if _status == 0 and os.path.isfile(l_max_pdb_file_name) and os.path.getsize(l_max_pdb_file_name):
                    _track_Lmaxima.append(local_maxima_log)
            if len(_track_Lmaxima) == len(sort_maxima[:-1]):
                _track_words += "done; "
            else:
                _track_words += "except for "
                for cc in [vv[0] for vv in sort_maxima[:-1]]:
                    if cc not in _track_Lmaxima:
                        _track_words += f"{cc} "
        
        logging.info("Done with current mol")
        logging.info(f"Energy barrier is: {barrier:.3f} /kcal.mol-1")
        if _dock:
            logging.info(f"Energy barrier for docking pose is: {dockP_ene:.3f} /kcal/mol-1")

        logging.info(_track_words)

        ## do clear
        
        os.system("rm -f _[A-Z]* *xyz xyz* *.ini traj*")

        if not os.path.exists("./Raw"):
            os.mkdir("./Raw")
        os.system("mv *.log ./Raw")
        os.system("mv *.gjf ./Raw")
        
    
    def result_2D_scan(self, scan_done_log):
        ## should have some kind of _Param.log2dih
        _log_dih = {}
        self.dih_param()
        ## there should be 2 param files
        _params = [pp for pp in os.listdir(".") if "_Param.log2dih" in pp]
        for log in scan_done_log:
            if "sp" not in log and "opt" not in log:
                _prefix = log.split(".")[0]
                with open(log, "r+") as ff:
                    content = [hh for hh in ff.readlines() if hh.strip()]
                
                validate_flag = [cc for cc in content if "Normal termination" in cc]
                
                if validate_flag:
                    ene = float([cc.strip() for cc in content if "SCF Done" in cc][0].split()[-5])
                    _dic_dih_each = {}
                    for each_pp in _params:
                        dih_label = int(each_pp.split(".")[-1])
                        get_dih_cmd = f"Multiwfn {log} < {each_pp} | grep 'The dihedral angle is' > _TMEP_{_prefix}_{dih_label}.dih"
                        (_status, _) = subprocess.getstatusoutput(get_dih_cmd)
                        if _status == 0 and \
                            os.path.isfile(f"_TMEP_{_prefix}_{dih_label}.dih") and \
                            os.path.getsize(f"_TMEP_{_prefix}_{dih_label}.dih"):
                            with open(f"_TMEP_{_prefix}_{dih_label}.dih", "r") as f:
                                line = [ff.strip() for ff in f.readlines()][0]
                            
                            _dih = Decimal(line.split()[-2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                            _dic_dih_each.setdefault(dih_label, _dih)

                    _sort_each = sorted(_dic_dih_each.items(), key=lambda x:x[0])
                    dih = [cc[-1] for cc in _sort_each]
                                          
                    _log_dih.setdefault(log, (dih, ene))
         ## get docking pose sp energy
        _dock = {}
        try:
            dockP_log = [cc for cc in scan_done_log if "sp" in cc][0]
        except Exception as e:
            logging.info("Something went wrong for docking pose scan, check if sp*.gjf exist")
        else:
            with open(dockP_log, "r+") as ff:
                content = [hh for hh in ff.readlines() if hh.strip()]
            
            validate_flag = [cc for cc in content if "Normal termination" in cc]

            if validate_flag:
                ene = float([cc.strip() for cc in content if "SCF Done" in cc][0].split()[-5])
                for each_pp in _params:
                    _dic_dih_each = {}
                    dih_label = int(each_pp.split(".")[-1])
                    get_dih_cmd = f"Multiwfn {dockP_log} < {each_pp} | grep 'The dihedral angle is' > _TMEP_dockP_{dih_label}.dih"
                    (_status, _) = subprocess.getstatusoutput(get_dih_cmd)
                    if _status == 0 and \
                    os.path.isfile(f"_TMEP_dockP_{dih_label}") and \
                    os.path.getsize(f"_TMEP_dockP_{dih_label}.dih"):
                        with open(f"_TMEP_dockP_{dih_label}", "r+") as f:
                            line = [ff.strip() for ff in f.readlines()][0]
                        
                        _dih = Decimal(line.split()[-2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                        _dic_dih_each.setdefault(dih_label, _dih)

                _sort_each = sorted(_dic_dih_each.items(), key=lambda x:x[0])
                
                dih = [cc[-1] for cc in _sort_each]
                _dock.setdefault("Docking Pose", (dih, ene))
        
        
        ## there you have _log_dih

        if _log_dih:
            _log_dih_sort = sorted(_log_dih.items(), key=lambda x: int(x[0].split(".")[0].split("_")[-1]))
            _ene_min = min([cc[-1] for cc in _log_dih.values()])
            
            _barrier_list = [(cc[-1][-1] - _ene_min)*627.51 for cc in _log_dih_sort]
            barrier = max(_barrier_list) - min(_barrier_list)

            X_dih1 = [cc[-1][0][0] for cc in _log_dih_sort]
            X_dih2 = [cc[-1][0][1] for cc in _log_dih_sort]

            ## process docking pose
            if _dock:
                dockP_ene = (_dock["Docking Pose"][-1] - _ene_min) * 627.51
                save_barrier =[dockP_ene] + _barrier_list
                save_dih1 = [_dock["Docking Pose"][0][0]] + X_dih1
                save_dih2 = [_dock["Docking Pose"][0][1]] + X_dih2
                save_indicator = ["Docking Pose"] + [cc[0].split(".")[0] for cc in _log_dih_sort]
            else:
                save_indicator = [cc[0].split(".")[0] for cc in _log_dih_sort]
                save_dih1 = X_dih1
                save_dih2 = X_dih2
                save_barrier = _barrier_list

            df = pd.DataFrame({"Pose/File Name": save_indicator, 
                               f"DIH:{self.bond1}/degree": save_dih1,
                               f"DIH:{self.bond2}/degree": save_dih2,
                               "Energy/kcal.mol-1": save_barrier})
            df.to_csv(f"Data_RigidScan_1D.csv", index=None)

        else:
            logging.info("Failed at get final result, check *.log file to see if no convergence meet")
            return

        ## graph and conf save
        unique_dih1 = []
        for i in range(len(X_dih1)):
            if X_dih1[i] not in unique_dih1:
                unique_dih1.append(X_dih1[i])
        unique_dih2 = []
        for i in range(len(X_dih2)):
            if X_dih2[i] not in unique_dih2:
                unique_dih2.append(X_dih2[i])
        
        dih_1_2 = []
        for i in range(len(unique_dih1)):
            for j in range(len(unique_dih2)):
                dih_1_2.append(tuple([unique_dih1[i], unique_dih2[j]]))

        zipped_dih_pair = list(zip(X_dih1, X_dih2))
        zipped_content = list(zip(X_dih1, X_dih2, _barrier_list))
        update_content = []

        for i in range(len(dih_1_2)):
            if dih_1_2[i] not in zipped_dih_pair:
                update_content.append(tuple([dih_1_2[i][0], dih_1_2[i][1], -0.1]))
            else:
                idx = zipped_dih_pair.index(dih_1_2[i])
                update_content.append(tuple([dih_1_2[i][0], dih_1_2[i][1], zipped_content[idx][-1]]))
        
        new_dih1, new_dih2, new_energy = zip(*(tuple(update_content)))

        new_dih1 = list(new_dih1)
        new_dih2 = list(new_dih2)
        new_energy = np.array(list(new_energy))

        upper_level = math.ceil(new_energy.max())

        torsion_angle_A = []
        for i in range(len(new_dih1)):
            if new_dih1[i] not in torsion_angle_A:
                torsion_angle_A.append(new_dih1[i])

        torsion_angle_B = []
        for i in range(len(new_dih2)):
            if new_dih2[i] not in torsion_angle_B:
                torsion_angle_B.append(new_dih2[i])
        
        plt.figure(figsize=(12,8), dpi=300)
        Z = new_energy.reshape(len(torsion_angle_A), len(torsion_angle_B))

        if upper_level > 50.0:
            level_basic = [i*0.1 for i in range(-1, 201)]
            level_extend = [20] + [i*2+0.5 for i in range(10, (upper_level // 2)+2)]
            a = plt.contourf(Z, levels=level_basic, cmap=plt.cm.rainbow)
            a1 = plt.contourf(Z, levels=level_extend, cmap=plt.cm.autumn)

            plt.colorbar(a)
            plt.colorbar(a1)
            
        else:
            level_basic = [i*0.1 for i in range(-1, upper_level*10+1)]
            a = plt.contourf(Z, levels=level_basic, cmap=plt.cm.rainbow)
            
            plt.colorbar(a)
        
        y_ticks = torsion_angle_A
        x_ticks = torsion_angle_B

        plt.xticks([i for i in range(len(x_ticks))], x_ticks)
        plt.yticks([i for i in range(len(y_ticks))], y_ticks)

        plt.ylabel(f"Dih1:{self.bond1}/degree")
        plt.xlabel(f"Dih2:{self.bond2}/degree")
        
        plt.savefig(f"Contour_RigidScan.png")

        ## find the  minimum and maximun
        minima = {}
        maxima = {}

        ##expand Z
        #pre_exp_Z = np.concatenate([Z, Z, Z], axis=1)
        #exp_Z = np.concatenate([pre_exp_Z, pre_exp_Z, pre_exp_Z], axis=0)
        #print(Z.shape, exp_Z.shape)
        i = 0
        while i < len(torsion_angle_A):
            j = 0
            if i-1 < 0:
                i_11 = -1
                i_12 = len(torsion_angle_A)
                i_21 = 0
                i_22 = i + 2
            elif i + 1 == len(torsion_angle_A):
                i_11 = i - 1
                i_12 = len(torsion_angle_A)
                i_21 = 0
                i_22 = 1
            else:
                i_11 = i - 1
                i_12 = i
                i_21 = i
                i_22 = i + 2

            while j < len(torsion_angle_B):
                if j - 1< 0:
                    j_11 = -1
                    j_12 = len(torsion_angle_B)
                    j_21 = 0
                    j_22 = j + 2
                elif j + 1 == len(torsion_angle_B):
                    j_11 = j - 1
                    j_12 = len(torsion_angle_B)
                    j_21 = 0
                    j_22 = 1
                else:
                    j_11 = j - 1
                    j_12 = j
                    j_21 = j
                    j_22 = j + 2
                
                current_scope = list(Z[i_11:i_12, j_11:j_12].flatten()) + \
                                list(Z[i_11:i_12, j_21:j_22].flatten()) + \
                                list(Z[i_21:i_22, j_11:j_12].flatten()) + \
                                list(Z[i_21:i_22, j_21:j_22].flatten())
                
                #print([vv for vv in dict(Counter(current_scope)).values()])
                
                if max([vv for vv in dict(Counter(current_scope)).values()]) > 1 or \
                   min(current_scope) < 0:
                   pass
                
                else:
                    if max(current_scope) == Z[i][j]:
                        maxima.setdefault(f"{torsion_angle_A[i]}_{torsion_angle_B[j]}", Z[i][j])
                    elif min(current_scope) == Z[i][j]:
                        minima.setdefault(f"{torsion_angle_A[i]}_{torsion_angle_B[j]}", Z[i][j])
                j += 1
            i += 1

        sort_maxima = sorted(maxima.items(), key=lambda x: x[1])
        sort_minima = sorted(minima.items(), key=lambda x: x[1])

        _track_words = ""

        _track_words += "get global minima "

        ## global minima
        global_minima = sort_minima[0]
        g_min_dih1, g_min_dih2 = global_minima[0].split("_")
        _get_minima_log = [kk for kk, vv in _log_dih.items() \
                            if vv[0][0] == int(g_min_dih1) and \
                            vv[0][1] == int(g_min_dih2)][0]
        g_min_pdb_file_name = f"Conf_GlobalMinima_dih1:{g_min_dih1}_dih2:{g_min_dih2}.pdb"
        conf_gen_param = ["100", "2", "1", g_min_pdb_file_name, "0", "q"]
        _status = self.gen_conf_Multiwfn(conf_gen_param, _get_minima_log)

        if _status == 0 and os.path.isfile(g_min_pdb_file_name) and os.path.getsize(g_min_pdb_file_name):
            _track_words += "done; "
        else:
            _track_words += "failed "
        
        if sort_minima[1:]:
            _track_words += "get local minima "
            _track_Lminima = []
            _error_Lminima = []
            for each in sort_minima[1:]:
                l_min_dih1, l_min_dih2 = each[0].split("_")
                _get_l_minima_log = [kk for kk, vv in _log_dih.items() \
                                        if vv[0][0] == int(l_min_dih1) and \
                                        vv[0][1] == int(l_min_dih2)][0]
                l_min_pdb_file_name = f"Conf_LocalMinima_dih1:{l_min_dih1}_dih2:{l_min_dih2}.pdb"
                conf_gen_param = ["100", "2", "1", l_min_pdb_file_name, "0", "q"]
                _status = self.gen_conf_Multiwfn(conf_gen_param, _get_l_minima_log)

                if _status == 0 and os.path.isfile(l_min_pdb_file_name) and os.path.getsize(l_min_pdb_file_name):
                    _track_Lminima.append(_get_l_minima_log)
                else:
                    _error_Lminima.append(_get_l_minima_log)


            if _error_Lminima:
                _track_words += "except for "
                for cc in _error_Lminima:
                     _track_words += f"{cc} "
            else:
                _track_words += "done; "
        
        _track_words += "get global maxima "

        global_maxima = sort_maxima[-1]
        g_max_dih1, g_max_dih2 = global_maxima[0].split("_")
        _get_maxima_log = [kk for kk, vv in _log_dih.items() \
                            if vv[0][0] == int(g_max_dih1) and \
                            vv[0][1] == int(g_max_dih2)][0]
        g_max_pdb_file_name = f"Conf_GlobalMaxima_dih1:{g_max_dih1}_dih2:{g_max_dih2}.pdb"
        conf_gen_param = ["100", "2", "1", g_max_pdb_file_name, "0", "q"]
        _status = self.gen_conf_Multiwfn(conf_gen_param, _get_maxima_log)

        if _status == 0 and os.path.isfile(g_max_pdb_file_name) and os.path.getsize(g_max_pdb_file_name):
            _track_words += "done; "
        else:
            _track_words += "failed "
        
        if sort_maxima[:-1]:
            _track_words += "get local maxima "
            _track_Lmax = []
            _error_Lmax = []
            for each in sort_maxima[:-1]:
                l_max_dih1, l_max_dih2 = each[0].split("_")
                _get_l_max_log = [kk for kk, vv in _log_dih.items() \
                                        if vv[0][0] == int(l_max_dih1) and \
                                        vv[0][1] == int(l_max_dih2)][0]
                l_max_pdb_file_name = f"Conf_LocalMaxima_dih1:{l_max_dih1}_dih2:{l_max_dih2}.pdb"
                conf_gen_param = ["100", "2", "1", l_max_pdb_file_name, "0", "q"]
                _status = self.gen_conf_Multiwfn(conf_gen_param, _get_l_max_log)

                if _status == 0 and os.path.isfile(l_max_pdb_file_name) and os.path.getsize(l_max_pdb_file_name):
                    _track_Lmax.append(_get_l_max_log)
                else:
                    _error_Lmax.append(_get_l_max_log)
            if _error_Lmax:
                _track_words += "except for "
                for cc in _error_Lmax:
                     _track_words += f"{cc} "
            else:
                _track_words += "done; "
        
        logging.info("Done with current mol")
        logging.info(f"Energy barrier is: {barrier:.3f} /kcal.mol-1")
        if _dock:
            logging.info(f"Energy barrier for docking pose is: {dockP_ene:.3f} /kcal/mol-1")

        logging.info(_track_words)

        ## do clear
        os.system("rm -f _[A-Z]* *xyz xyz* *.ini traj*")

        if not os.path.exists("./Raw"):
            os.mkdir("./Raw")
        os.system("mv *.log ./Raw")
        os.system("mv *.gjf ./Raw")
    
    def dih_param(self):
        atom_list1 = [f"a{i}" for i in (self.bond1.strip()).split(",")]
        content = ["2", "-9", " ".join(atom_list1), "q", "-10", "q"]
        with open("_Param.log2dih.1", "w+") as cc:
            for line in content:
                cc.write(line + "\n")

        if self.bond2:
            ## should be double
            atom_list2 = [f"a{i}" for i in (self.bond2.strip()).split(",")]
            content2 = ["2", "-9", " ".join(atom_list2), "q", "-10", "q"]
            with open("_Param.log2dih.2", "w+") as cc2:
                for line in content2:
                    cc2.write(line + "\n")
        
        return 
        

    def proceed(self):
        ## initial run
        logging.info("STEP1: Try to initialize system and run optimization")
        try:
            opt_run_done_log = self.run_initial()
        except Exception as e:
            logging.info("Termination at optimization stage")
            return

        if not opt_run_done_log:
            logging.info("Termination at optimization stage")
            return

        logging.info("STEP2: Try to do rotation scan")
        try:
            sp_run_don_log = self.run_rotation(opt_run_done_log)
        except Exception as e:
            logging.info("Termination at rotation scan stage")
            return 
        if not sp_run_don_log:
            logging.info("Termination at rotation scan stage")
            return

        logging.info("STEP3: Try to do collect result")
        if self.bond2:
            self.result_2D_scan(sp_run_don_log)
        else:
            self.result_1D_scan(sp_run_don_log)
        
        return



    
    

