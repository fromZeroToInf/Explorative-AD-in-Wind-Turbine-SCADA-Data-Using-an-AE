import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import re
import json
from enum import StrEnum
from helperfunctions import intern_constants as ic
from typing import TypedDict, List

class PreProcKeys(StrEnum):
    VERSION = "version"
    TRAIN_START = "Train_Start"
    TRAIN_END = "Train_End"
    IMPUTED_PATH = "imputed_path"
    CLEANED_DATA_PATH = "cleaned_data_path"
    MINMAX_FILENAME = "minmax_filename"
    PC_MASKS_PATH = "pc_masks_path"
    IDIO_PATH = "idio_path"
    FM_PATH = "fm_path"
    EXCLUDECOLLIST = "excludeColList"
    
class PreProcDict(TypedDict):
    version: str
    Train_Start: str 
    Train_End: str
    imputed_path: str
    cleaned_data_path: str 
    minmax_filename: str
    pc_masks_path: str
    idio_path: str
    fm_path: str
    excludeColList: List[str]
class PreprocessingStep5():
    
    def _create_ref_powercurve(self):
        ### The Powercurve data for the senvion mm82 2050 was extracted from: https://www.thewindpower.net/turbine_en_464_senvion_mm82-2050.php
        ### (In the internet exists at least 3 different power curves for RePower/senvion mm82 2050)
        ###The implemented powercurve has been adjusted. The cut-off is now at 25 m/s due to inconsistencies with the observed data.
        pc_dir = Path(ic.PATH_PENMANSHIEL) / "powercurve"
        pc_dir.mkdir(parents=True, exist_ok=True)

        pc_path = pc_dir / "powercurve.csv"

        data = [
            (4,55),
            (4.5,110),
            (5,186),
            (5.5,264),
            (6,342),
            (6.5,424),
            (7,506),
            (7.5,618),
            (8,730),
            (8.5,865),
            (9,999),
            (9.5,1195),
            (10,1391),
            (10.5,1558),
            (11,1724),
            (11.5,1829),
            (12,1909),
            (12.5,1960),
            (13,2002),
            (13.5,2025),
            (14,2044),
            (14.5,2050),
            (15,2050),
            (15.5,2050),
            (16,2050),
            (16.5,2050),
            (17,2050),
            (17.5,2050),
            (18,2050),
            (18.5,2050),
            (19,2050),
            (19.5,2050),
            (20,2050),
            (20.5,2050),
            (21,2050),
            (21.5,2050),
            (22,2050),
            (22.5,2050),
            (23,2050),#adjustement
            (23.5,2050),#adjustement
            (24,2050),#adjustement
            (24.5,2050),#adjustement
            (25,2050),#adjustement
        ]

        df_pc_raw = pd.DataFrame(data, columns=["Wind speed (m/s)", "Power (kW)"])
        df_pc = df_pc_raw.copy()
        df_pc["Wind speed (m/s)"] = df_pc["Wind speed (m/s)"].astype(float)
        df_pc["Power (kW)"] = df_pc["Power (kW)"].astype(float)
        df_pc = df_pc.sort_values("Wind speed (m/s)").reset_index(drop=True)
        df_pc.to_csv(pc_path, index=False)
    
        #################################################
        # This code section is from https://github.com/sltzgs/OpenWindSCADA/blob/main/notebooks/helper_functions.py and was adapted. 
    def _prepare_power_curve(self)-> pd.DataFrame:
        pc_path =os.path.join(ic.PATH_PENMANSHIEL, "powercurve")
        f_path = os.path.join(pc_path,"powercurve.csv")
        if not os.path.exists(f_path):
            os.makedirs(pc_path, exist_ok=True)
            self._create_ref_powercurve()
        
        df_powercurve = pd.read_csv(f_path, index_col=0)
        df_pc = pd.DataFrame(index=[np.round(i, 2) for i in np.arange(0, 30, 0.01)], columns=['power_norm'])
        df_pc[df_pc.index < df_powercurve.index[0]] = 0
        df_pc[df_pc.index > df_powercurve.index[-1]] = 0
        df_pc.loc[df_powercurve.index] = df_powercurve.values.reshape(-1, 1)
        return df_pc.astype(float).interpolate(method='linear')
    ##################################################
    
    def _nb_filtering_by_powercurve(
        self,
        df:pd.DataFrame, 
        df_pc:pd.DataFrame, 
        wind_col:str="Wind speed (m/s)", 
        power_col:str="Power (kW)", 
        min_wind_speed: float = 0.0,
        max_wind_speed: float = 25.0,
        *,
        max_abs_margin_kw: float = 0, #(Senvion MM82B rated power 2050 kW)
        )-> pd.Series:
        
        if df.index.name != ic.TS_COL and ic.TS_COL in df.columns:
            df = df.set_index(ic.TS_COL)
            df.index.name = ic.TS_COL
        
        df = df.sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None) #safeguard
        
        
        mask_wind = ((df[wind_col] >= min_wind_speed) & 
                    (df[wind_col] <= max_wind_speed))
        
        
        wind_speed = np.round(df[wind_col],2)
        valid = df.index[mask_wind]
        exp_power = pd.Series(index=df.index, dtype=np.float64)
        exp_power.loc[valid] = df_pc.loc[wind_speed.loc[valid].values, "power_norm"].values
        
        margin = max_abs_margin_kw
        
        mid = exp_power.values
        lower_bound = mid - margin
        upper_bound = mid +margin
        
        lower_bound = np.where( lower_bound < 0.0, 0.0, lower_bound)
        
        mask_power = (df[power_col].values >= lower_bound) & (df[power_col].values <= upper_bound)
        
        return pd.Series(mask_wind & mask_power, index=df.index, name="pc_mask")
    
    
    def _fleetmedian(self, df_fleet:pd.DataFrame)-> pd.DataFrame:
        df_fm = (
        df_fleet
        .groupby(lambda col: col.rsplit("_",1)[0],axis=1, sort=False)
        .median() 
        )
        return df_fm

    def _idiosyncratic_components(self, df_fleet: pd.DataFrame) -> pd.DataFrame:
        df_fleet = df_fleet.copy()
        base = df_fleet.columns.to_series().apply(lambda c: c.rsplit("_",1)[0])
        
        df_fm = df_fleet.groupby(base, axis=1, sort=False).median()
        
        df_fm_expanded = df_fm.reindex(columns=base, copy=False)
        
        df_idio = df_fleet - df_fm_expanded.values
        df_idio.columns = df_fleet.columns
        df_idio.index = df_fleet.index
        return df_idio

    def _PreStep5_update_to_df( self,
                                config: PreProcDict,
                                files:list,
                                df_fleet_median: pd.DataFrame,
                                df_idiosyncratic_components: pd.DataFrame)-> None:
        
        os.makedirs(config["idio_path"], exist_ok=True)
        os.makedirs(config["fm_path"], exist_ok=True)
        
        for f in tqdm(files):
            base = os.path.basename(f)

            df_origin = (
                pd.read_csv(f, parse_dates=[ic.TS_COL])
                .set_index(ic.TS_COL)
            )
            
            df_origin = df_origin.copy()
            df_origin.index = pd.to_datetime(df_origin.index).tz_localize(None) # safeguard
            df_origin = df_origin.sort_index()
            
            wt_id = str(df_origin[ic.WT_ID].iat[0])
            
            wt_cols = [c for c in df_idiosyncratic_components.columns if c.endswith(f"_{wt_id}")]
            df_idio = df_idiosyncratic_components[wt_cols].copy()
            
            df_idio.columns = [c.rsplit("_",1)[0] for c in wt_cols]
            df_origin = df_origin.loc[config["Train_Start"]: config["Train_End"]]
            
            
            df_idio.index = pd.to_datetime(df_idio.index).tz_localize(None)
            df_idio = df_idio.reindex(df_origin.index)

            df_origin.update(df_idio)
        
            
            fname = os.path.join(config["idio_path"], base.removesuffix(".csv") + "_idio_comps.csv")
            df_origin = df_origin.reset_index()
            df_origin.to_csv(fname, index=False)
            #df_origin.head()
            fname_fleetm = os.path.join(config["fm_path"], "fleet_median.csv")
            df_fleet_median.to_csv(fname_fleetm, index_label=ic.TS_COL)
        
        
    def _PrepStep5_fleetmedian(self, config: PreProcDict, load_data_path: str, freq: str = "10min")-> None:
        
        files = glob(os.path.join(load_data_path, "*.csv"))
        dfs = []
        for f in tqdm(files):
            df = pd.read_csv(f, parse_dates=[ic.TS_COL])
            
            df_cpy = df.copy()
            df_cpy = df_cpy.set_index(ic.TS_COL)
            
            df_cpy.index = pd.to_datetime(df_cpy.index).tz_localize(None) #safeguard
            df_cpy = df_cpy.sort_index()
            
            
            wt_id = str(df_cpy[ic.WT_ID].iat[0])
            df_cpy = df_cpy.drop(columns=[ic.WT_ID])
            
            start, end = df_cpy.index.min(), df_cpy.index.max()
            
            if (end - start).days < 365:
                print(f"The training time period is less then 1 year")
                #raise ValueError(f"The training time period needs to be at least 1 year")

            df_sub = df_cpy.loc[config["Train_Start"]: config["Train_End"]]
            df_sub = df_sub[[col for col in df.columns if col not in config["excludeColList"] ]]
            df_sub.columns = [f"{col}_{wt_id}" for col in df_sub.columns]
            dfs.append(df_sub)
        
            
        df_fleet = pd.concat(dfs, axis=1, join="outer").sort_index()
        
        df_fleet_median = self._fleetmedian(df_fleet)
        
        df_idiosyncratic_components = self._idiosyncratic_components(df_fleet)
        
        if df_fleet_median.isna().any().any():
            nan_locs = df_fleet_median.isna()
            first_idx = nan_locs.stack().index[nan_locs.stack()].tolist()[:3]
            raise ValueError(f"Fleet-Median contains NaN values! E.g. {first_idx}"
                            f"Something went wrong. Please check!")
        
        if df_idiosyncratic_components.isna().any().any():
            nan_locs = df_idiosyncratic_components.isna()
            first_idx = nan_locs.stack().index[nan_locs.stack()].tolist()[:3]
            raise ValueError(f"Idio-Comps contain NaN values. E.g. {first_idx}"
                            f"Check Indices (TS) and creation of median...")
        
        self._PreStep5_update_to_df(config, files, df_fleet_median, df_idiosyncratic_components)
    
    
    def _do_pc_masks(
                self,
                config: PreProcDict,
                load_data_path: str,
                training:bool=True
                )->None:
        files = glob(os.path.join(load_data_path, "*.csv"))
        os.makedirs(config["pc_masks_path"], exist_ok=True)
        
        df_pc = self._prepare_power_curve()
        for f in tqdm(files, desc="creating pc masks"):
            df = pd.read_csv(f).set_index(ic.TS_COL)
            
            df = df.sort_index()
            df.index = pd.to_datetime(df.index).tz_localize(None) #safeguard
            
            df_sub = df.copy().loc[config["Train_Start"]: config["Train_End"]]
            
            pc_mask = self._nb_filtering_by_powercurve(df_sub, df_pc, max_abs_margin_kw=150, min_wind_speed=5.0)
            
            if training:
                fname = ("WT_ID_"+ os.path.basename(f).removesuffix(".csv").split("_")[2] + 
                        f"_{config['version']}_training"
                        f"_pc_mask.csv")
            else:
                fname = ("WT_ID_"+ os.path.basename(f).removesuffix(".csv").split("_")[2] + 
                        f"_{config['version']}_val_test"
                        f"_pc_mask.csv")
            fname = os.path.join(config["pc_masks_path"], fname)
            
            pc_mask.to_frame().reset_index().to_csv(fname, index=False)
    
    def _get_pc_filter(self,config: PreProcDict, wt_id: str)-> pd.Series:
        files = glob(os.path.join(config["pc_masks_path"], "*.csv"))
        pattern = re.compile(r"^WT_ID_(\d+)")
        
        for f in tqdm(files):
            fn = os.path.basename(f)
            m = pattern.match(fn)
            if m and int(float(m.group(1))) == int(wt_id):
                df = pd.read_csv(f, parse_dates=[ic.TS_COL]).set_index(ic.TS_COL)
                df = df.sort_index()
                msk = df["pc_mask"].astype(bool)
                msk.index = pd.to_datetime(msk.index).tz_localize(None) # safeguard
                return msk
        raise FileNotFoundError(f"power curve mask does not exist for WT_ID ={wt_id} in folder {config["pc_masks_path"]} ")
    
    def _fit_minmax_scalers_from_pc_masks(self, config: PreProcDict) -> None:
        files = glob(os.path.join(config["idio_path"], "*.csv"))
        
        if not files:
            raise ValueError(f"No files found in {config["idio_path"]}.")
        
        feature_cols = [c for c in pd.read_csv(files[0]).columns if c not in [ic.WT_ID, ic.TS_COL]]
        
        scalers: dict[int, MinMaxScaler] = {} # a MinMaxScaler per WT
        for f in tqdm(files, desc="Fit MinMax per WT on idio. comps."):
            df_idio = pd.read_csv(f, parse_dates=[ic.TS_COL]).set_index(ic.TS_COL).sort_index()
            df_idio.index = pd.to_datetime(df_idio.index).tz_localize(None)
            
            wt_id = int(df_idio[ic.WT_ID].iloc[0])
            
            mask = self._get_pc_filter(config, wt_id=wt_id).reindex(df_idio.index, fill_value=False)
            
            X_fit = df_idio.loc[mask.values, feature_cols].to_numpy(dtype=float, copy=False)
            if X_fit.size == 0:
                raise ValueError(f"WT {wt_id}: No rows from df_train of pc mask to fit the scaler")
            
            scaler = MinMaxScaler().fit(X_fit)
            scalers[wt_id] = scaler
            
        joblib.dump(scalers, config["minmax_filename"])
        os.makedirs(ic.PATH_FEATURE_ORDER, exist_ok=True)
        
        Path(ic.PATH_FEATURE_ORDER_FILE).write_text(json.dumps(feature_cols, ensure_ascii=False), encoding="utf-8")
    
    def _save_pc_filtered_train_csvs(
        self,
        config: PreProcDict, 
        source_dir= str
        ) -> None:
        """ Files saved at config["cleaned_data_path"]. See config.
        """
        files = glob(os.path.join(source_dir, "*.csv"))
        os.makedirs(config["cleaned_data_path"], exist_ok=True)
        
        for f in tqdm(files, desc=f"saving PC-filtered Train CSVs in {config['cleaned_data_path']}"):
            df = pd.read_csv(f, parse_dates=[ic.TS_COL]).set_index(ic.TS_COL).sort_index()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            df_train = df.loc[config["Train_Start"]: config["Train_End"]]
            if df_train.empty:
                raise ValueError(f"{os.path.basename(f)}: no rows in df_train")
            
            wt_id = int(df_train[ic.WT_ID].iloc[0])
            
            pc_mask = self._get_pc_filter(config, wt_id=wt_id).reindex(df_train.index, fill_value=False)
            
            df_filtered = df_train[pc_mask.values]
            if df_filtered.empty:
                raise ValueError(f"WT={wt_id}: No rows exist after PC-filtering in train time period.")
            
            file_name = f"WT_ID_{wt_id}_{config['version']}_training_pc_filtered.csv"
            
            file_path = os.path.join(config["cleaned_data_path"],file_name)
            df_filtered.reset_index().to_csv(file_path, index=False)
    
        
    def execute_pre_step5(self, config: PreProcDict) -> None:
        self._do_pc_masks(config, config["imputed_path"])
        # writes *_idio_comps_csvs to path config["idio_path"]
        # and fleet_median.csv to path config["fm_path"]
        self._PrepStep5_fleetmedian(config, load_data_path=config["imputed_path"])
        self._fit_minmax_scalers_from_pc_masks(config)
        self._save_pc_filtered_train_csvs(config, source_dir= config["imputed_path"])