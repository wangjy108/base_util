#!/usr/bin/env python
# coding: utf-8

"""
// author: Wang (Max) Jiayue 
// email: wangjy108@outlook.com
// git profile: https://github.com/wangjy108
"""

import os
import sys
import pandas as pd
from rdkit import Chem
import argparse
import yaml
import logging
import Auto3D
from Auto3D.auto3D import options, main

#logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.WARNING)

"""
// input smi, generate sdf
"""

class ConfGen():
    def __init__(self,**args):
        self.smi = args["input_smi_file"]

        #self.platform = args["platform"]

        self.path = os.path.join(os.getcwd(), "_input.smi")

        try:
            self.k = args["k"]
        except Exception as e:
            self.k = None
        else:
            try:
                self.k = int(self.k)
            except Exception as e:
                self.k = None
        
        if not self.k:
            try:
                self.window = args["window"]
            except Exception as e:
                self.window = 20.0
            else:
                try:
                    self.window = float(self.window)
                except Exception as e:
                    self.window = 20.0
        else:
            self.window = None
        
        self.memory = None
        self.capacity = 40
         
        try:
            self.enumerate_tautomer = args["enumerate_tautomer"]
        except Exception as e:
            self.enumerate_tautomer = False
        else:
            if not isinstance(self.enumerate_tautomer, bool):
                self.enumerate_tautomer = False
        
        self.tauto_engine = 'rdkit'
        self.pKaNorm = True
        self.isomer_engine = 'rdkit'
        self.enumerate_isomer = True
        self.mode_oe = 'classic'

        try:
            self.mpi_np = args["mpi_np"]
        except Exception as e:
            self.mpi_np = 8
        else:
            try:
                self.mpi_np = int(self.mpi_np)
            except Exception as e:
                self.mpi_np = 8
            else:
                if self.mpi_np <= 0:
                    self.mpi_np = 8

        try:
            self.optimizing_engine = args["optimizing_engine"]
        except Exception as e:
            self.optimizing_engine = 'AIMNET'
        else:
            if self.optimizing_engine not in ['ANI2x', 'AIMNET']:
                self.optimizing_engine = 'AIMNET'
        
        try:
            self.use_gpu = args["use_gpu"]
        except Exception as e:
            self.use_gpu = False
        else:
            if not isinstance(self.use_gpu, bool):
                self.use_gpu = False
        
        if self.use_gpu:
            self.gpu_idx = 0
        else:
            self.gpu_idx = None

        self.opt_steps = 5000
        self.convergence_threshold = 0.0001
        self.patience = 1000

        try:
            self.threshold = args["threshold"]
        except Exception as e:
            self.threshold = 0.5
        else:
            try:
                self.threshold = float(self.threshold)
            except Exception as e:
                self.threshold = 0.5
            else:
                if self.threshold <= 0:
                    self.threshold = 0.5
        
        self.verbose = False
        self.job_name = "_input"
        self.pKaNorm = False

        self.arguments = options(
            self.path,
            k=self.k,
            window=self.window,
            verbose=self.verbose,
            job_name=self.job_name,
            enumerate_tautomer=self.enumerate_tautomer,
            tauto_engine=self.tauto_engine,
            pKaNorm=self.pKaNorm,
            isomer_engine=self.isomer_engine,
            enumerate_isomer=self.enumerate_isomer,
            mode_oe=self.mode_oe,
            mpi_np=self.mpi_np,
            #max_confs=self.max_confs,
            use_gpu=self.use_gpu,
            gpu_idx=self.gpu_idx,
            capacity=self.capacity,
            optimizing_engine=self.optimizing_engine,
            opt_steps=self.opt_steps,
            convergence_threshold=self.convergence_threshold,
            patience=self.patience,
            threshold=self.threshold,
            memory=self.memory
        )

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
        
        format_smi = pd.DataFrame({"smi":self.input_smi, \
                                   "idx": [f"M{ii}" for ii in range(len(self.input_smi))]})
        format_smi.to_csv(os.path.join(os.getcwd(),"_input.smi"), header=None, index=None, sep='\t')


    def run(self):
        """
        run_dict{"lbg": , run_lbg, \
                "local": run_local}
        
        try:
            run_dict[self.platform]
        except Exception as e:
            logging.info(f"Wrong platform setting, please choose from [{[kk for kk in run_dict.keys()]}]")
            return None
        """
        
        out = main(self.arguments)
        
    


    







