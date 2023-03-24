#!/usr/bin/env python
# coding: utf-8

"""
// author: Wang (Max) Jiayue 
// email: wangjy108@outlook.com
// git profile: https://github.com/wangjy108
"""

from rdkit import rdBase, Chem
from rdkit.Chem import Draw, AllChem
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

"""
// input smi, generate sdf
// if more than one conformer save herein, 
    align reference is could either be lowest RMSD conformer,
    or conformer with lowest energy tag
"""

class ConfGen():
    def __init__(self, **args):
        self.smi = args["input_smi_file"]
        self.method = args["method"]
        self.serial_name = self.smi.split("/")[-1].split(".")[0]

        try:
            self.save_file_name = os.path.join(os.getcwd(), args["fileName"])
        except Exception as e:
            self.save_file_name = os.path.join(os.getcwd(), "SAVE.sdf")
        
        try:
            self.genConfNum = args["genConfNum"]
        except Exception as e:
            self.genConfNum = 20
        
        try:
            self.saveConfNum = args["saveConfNum"]
        except Exception as e:
            self.saveConfNum = 20

        ##automatic define if there is header in input smi csv file
        df = pd.read_csv(self.smi, sep="\\s+", header=None)
        df_header = pd.read_csv(self.smi, sep="\\s+")
        if list(df_header.columns)[0] == df.iloc[:,0].to_list()[0]:
            self.df = df
        else:
            self.df = df_header

        ##automatic define which column contain smile string
        ii = 0
        while ii < self.df.shape[-1]:
            test_type_first = self.df.iloc[0,ii]
            try:
                Chem.MolFromSmiles(str(test_type_first))
            except Exception as e:
                ii += 1
            else:
                self.input_smi = self.df.iloc[:, ii].to_list()
                break
    
    def GenByMMFF94(self, inputSMI:list, genConfNum:int, saveConfNum:int, save_file_name:str, DBprefix:str):
        """
        gen3D mol by rdkit
        optimize by mm force filed mmff94, return with conformation set obj
        =====
        input: inputfile name;
        args: 
            //fileName: saved sdf file name
            //genConfNum: number of conformation generate for each smi;
            //saveConfNum: number of conformation to save;
        """

        cc = Chem.SDWriter(save_file_name)

        for i in range(len(inputSMI)):
            mol = Chem.MolFromSmiles(inputSMI[i])
            #mol = input_mol[i]
            m3d = Chem.AddHs(mol)
            molNameLabel = f"{DBprefix}_{i}"

            try:
                nGenConfs = AllChem.EmbedMultipleConfs(m3d,numConfs=genConfNum, numThreads=0)
            except Exception as e:
                #wrong_smi.append(inputSmi[i])
                continue
            else:
                if len(nGenConfs) == 0:
                    #wrong_smi.append(inputSmi[i])
                    continue

            res = AllChem.MMFFOptimizeMoleculeConfs(m3d, numThreads=0)

            if len(res) == 0:
                #wrong_smi.append(inputSmi[i])
                continue
            elif min([rr[0] for rr in res]) < 0:
                m3d.SetProp("cSMILES", inputSMI[i])
                m3d.SetProp("_Name", molNameLabel)
                #m3d.SetProp("conf_idx", "0")
                #m3d.SetProp("MM_energy", "0.0")
                cc.write(m3d, confId=0)
                continue

            elif len([(i, res[i][-1]) for i in range(len(res)) if res[i][0] == 0]) < saveConfNum:
                stable_conf = [(i, res[i][-1]) for i in range(len(res)) if res[i][0] == 0] + \
                            sorted([(i, res[i][-1]) for i in range(len(res)) if res[i][0] == 1], \
                            key=lambda x: x[-1], reverse=False)[:saveConfNum-len([(i, res[i][-1]) \
                            for i in range(len(res)) if res[i][0] == 0])]
            else:
                stable_conf = sorted([(i, res[i][-1]) for i in range(len(res))],\
                            key=lambda x: x[-1], reverse=False)[:saveConfNum]


            for ii in stable_conf:
                m3d.SetProp("cSMILES", inputSMI[i])
                m3d.SetProp("_Name", molNameLabel)
                #m3d.SetProp("conf_idx", str(ii[0]))
                #m3d.SetProp("MM_energy", f"{ii[1]:.7f}")
                cc.write(m3d, confId=ii[0])

        cc.close()

        try:
            Chem.SDMolSupplier(save_file_name)
        except Exception as e:
            os.system(f"rm -f {save_file_name}")
            return inputSMI

        sdf = set([mm.GetProp("cSMILES") for mm in Chem.SDMolSupplier(save_file_name) if mm])

        wrong_smi = set([ii for ii in inputSMI if ii not in sdf])

        return list(wrong_smi)
    
    def run(self):
        run_dict = {"MMFF94": self.GenByMMFF94}

        try:
            run_dict[self.method]
        except Exception as e:
            logging.info(f"Wrong method setting, please choose from {[kk for kk in run_dict.keys()]}")
            return 
        
        left_smi = run_dict[self.method](self.input_smi, \
                                         self.genConfNum, \
                                         self.saveConfNum, \
                                         self.save_file_name,\
                                         self.serial_name)

        logging.info(f"Generated confortmation saved in {self.save_file_name}")

        if left_smi:
            ee = pd.DataFrame({"":left_smi})
            ee.to_csv("ERROR.smi", header=None, index=None)
            logging.info("Not generated smiles are saved in ERROR.smi")

        return 



