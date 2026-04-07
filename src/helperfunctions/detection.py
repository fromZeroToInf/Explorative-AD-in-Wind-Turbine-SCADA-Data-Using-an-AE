from helperfunctions import intern_constants as ic
from helperfunctions.helper import load_feature_order as lfo
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Literal, Dict, Any
import numpy as np
import os
from enum import StrEnum
from scipy.stats import ks_2samp
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from dataclasses import dataclass
import matplotlib.pyplot as plt

class NBM_COLS(StrEnum):
    SIG_NAME = "signal_name"
    WIND_BIN_CENTER = "wind_bin_center"
    N_SAMPLES = "n_samples"
    MEAN_VALUE = "mean_value"
    STD_VALUE = "std_value"

class DET_SIG_COLS(StrEnum):
    TS_DETECTION = ic.TS_COL
    SIGNAL_NAME = "signal_name"
    WINDOW_START = "window_start"
    WINDOW_END = "window_end"
    WIND_CENTER = "wind_center"
    WIND_HALF_WIDTH = "wind_half_width"
    
    N_BASELINE = "n_baseline"
    MEAN_BASELINE = "mean_baseline"
    STD_BASELINE = "std_baseline"
   
    
    N_EVENT = "n_event"
    MEAN_EVENT = "mean_event"
    STD_EVENT = "std_event"
    
    VALUE_AT_TS = "value_at_ts"
    DELTA_AT_TS = "delta_at_ts"
    Z_AT_TS = "z_at_ts"
    DELTA_MEAN = "delta_mean"
    Z_SHIFT = "z_shift"
    RE_AT_TS = "re_at_ts"
    

class DET_OVERVIEW_COLS(StrEnum):
    TS_DETECTION = ic.TS_COL
    WINDOW_START = "window_start"
    WINDOW_END = "window_end"
    RE_MEAN = "re_mean"
    
    WIND_AT_TS = "wind_at_ts"
    POWER_AT_TS = "power_at_ts"
    PC_POWER_AT_TS = "pc_power_at_ts"
    RESIDUAL_AT_TS = "residual_at_ts"
    SHARE_WITHIN_PC_BAND_WINDOW = "share_within_pc_band_window"
    SHARE_BELOW_PC_BAND_WINDOW = "share_below_pc_band_window"
    SHARE_ABOVE_PC_BAND_WINDOW = "share_above_pc_band_window"
    
    N_TOP_SIGNALS = "n_top_signals"
    
    TAG_TEMP_HIGH = "tag_temp_high"
    TAG_VIBRATION_HIGH = "tag_vibration_high"
    TAG_PITCH_ASYMMETRY = "tag_pitch_asymmetry"
        
class Part2:
    
    @staticmethod
    def load_threshold_table() -> float:
        fp_best_k = ic.PATH_PART1_K_AGG_METRICS_DIR / "best_ks_per_wt.csv"
        threshold_df = pd.read_csv(fp_best_k)
        print(threshold_df.head())
        theta = threshold_df["threshold"].iloc[0]
        return theta
    
    @staticmethod
    def drop_imputations(
    df_eval: pd.DataFrame,
    pc: bool = False,
    power_col:str = "Power (kW)",
    wind_col: str = "Wind speed (m/s)",
    time_step: pd.Timedelta = pd.Timedelta(minutes=10),
    )-> pd.DataFrame:
    
        df = df_eval.copy()
        df[ic.TS_COL] = pd.to_datetime(df[ic.TS_COL], errors="raise")
        path_impute = Path(ic.PATH_IMPUT_MASKS)
        parquet_files = list(path_impute.glob("*.parquet"))
        
        t_min = df[ic.TS_COL].min()
        t_max = df[ic.TS_COL].max()
        wts = df[ic.WT_ID].unique()
        
        if pc:
            missing_pc = [c for c in (power_col, wind_col) if c not in df.columns]
            if missing_pc:
                raise KeyError(f"Power or Wind speed is missing in df")
            
        impute_frames: list[pd.DataFrame] = []
        
        signal_any = getattr(ic, "ANY", "ANY")
        
        for fp in parquet_files:
            df_impute = pd.read_parquet(fp)
            
            df_impute[ic.TS_COL] = pd.to_datetime(df_impute[ic.TS_COL], errors="raise")
            df_impute = df_impute.dropna(subset=[ic.TS_COL])
            
            if "Signal" in df_impute.columns:
                df_impute = df_impute[df_impute["Signal"].isin([signal_any])]
            
            df_impute = df_impute[
                df_impute[ic.WT_ID].isin(wts)
                & (df_impute[ic.TS_COL] >= t_min)
                & (df_impute[ic.TS_COL] <= t_max)
            ]
            
            if df_impute.empty:
                continue
            
            impute_frames.append(df_impute[[ic.WT_ID, ic.TS_COL]])
        
        if not impute_frames:
            return df
        
        flagged = (
            pd.concat(impute_frames, ignore_index=True)
            .dropna(subset=[ic.WT_ID, ic.TS_COL])
            .drop_duplicates([ic.WT_ID, ic.TS_COL])
        )
        
        flagged[ic.WT_ID] = flagged[ic.WT_ID].astype(df[ic.WT_ID].dtype)
        flagged[ic.TS_COL] = pd.to_datetime(flagged[ic.TS_COL], errors="raise")
        
        spans: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
        flagged_sorted = flagged.sort_values([ic.WT_ID, ic.TS_COL])
        for wt, g in flagged_sorted.groupby(ic.WT_ID, sort=False):
            ts_list = pd.to_datetime(g[ic.TS_COL]).drop_duplicates().sort_values().tolist()
            if not ts_list:
                continue
            
            run_start = ts_list[0]
            prev = ts_list[0]
            
            for cur in ts_list[1:]:
                if(cur-prev)==time_step:
                    prev =cur
                    continue
                spans.append((pd.Timestamp(run_start), pd.Timestamp(prev + time_step)))
                run_start =cur
                prev = cur
            
            spans.append((pd.Timestamp(run_start), pd.Timestamp(prev + time_step)))
        
        if not spans:
            return df
        
        t_max_excl = pd.to_datetime(t_max) + time_step
        spans_clipped: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in spans:
            ss = max(pd.to_datetime(s), pd.to_datetime(t_min))
            ee = min(pd.to_datetime(e), t_max_excl)
            if ss < ee:
                spans_clipped.append((ss,ee))
        
        if not spans_clipped:
            return df
        
        spans_clipped.sort(key=lambda x: x[0])
        merged_spans: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur_s, cur_e = spans_clipped[0]
        for s, e in spans_clipped[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged_spans.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged_spans.append((cur_s, cur_e))
        
        ts_ns = df[ic.TS_COL].to_numpy(dtype="datetime64[ns]").astype("int64")
        start_ns = np.array([s.value for s, _ in merged_spans], dtype=np.int64)
        ends_ns = np.array([e.value for _, e in merged_spans], dtype=np.int64)
        
        idx= np.searchsorted(start_ns, ts_ns, side="right") - 1
        in_span = np.zeros(len(ts_ns), dtype=bool)
        valid = idx >= 0
        in_span[valid] = ts_ns[valid] < ends_ns[idx[valid]]
        
        if not pc:
            drop_mask = in_span
        else:
            drop_mask = in_span & df[power_col].isna() & df[wind_col].isna()
        
        df_filtered = df.loc[~drop_mask].reset_index(drop=True)
        
        return df_filtered 
    
    @staticmethod
    def select_top_detections_with_gap(
    det_df: pd.DataFrame,
    gap_days: int,
    top_n: Optional[int] = None,
    save_filename: str = "part2_detections.csv",
    ) -> pd.DataFrame:
        """
        Detections are only selected if they have at least a distance of gap_days.
        Saves dataframe to ic.PATH_PART2_DETECTIONS / "part2_detections.csv".
        Returns:
            pd.DataFrame: cols are same as det_df
        """
        gap = pd.Timedelta(days=gap_days)
        
        det_df = det_df.copy()
        det_df[ic.TS_COL] = pd.to_datetime(det_df[ic.TS_COL], errors="raise")
        
        det_sorted = det_df.sort_values(by=ic.MEAN_LOSS_PER_SAMPLE, ascending=False).reset_index(drop=True)
        
        selected_idx : List[int] = []
        selected_ts_per_wt: dict[int, List[pd.Timestamp]] = defaultdict(list)
        
        for idx, row in det_sorted.iterrows():
            ts = row[ic.TS_COL]
            wt = int(row[ic.WT_ID])
            
            prev_ts_list= selected_ts_per_wt[wt]
            
            if not prev_ts_list:
                take = True
            else:
                diff = (ts - pd.Series(prev_ts_list)).abs()
                take = (diff >= gap).all()
                
            if take:
                selected_idx.append(idx)
                selected_ts_per_wt[wt].append(ts)
                
                if top_n is not None:
                    if len(selected_idx) >= top_n:
                        break
        
        selected_detections = det_sorted.loc[selected_idx].reset_index(drop=True)
        os.makedirs(ic.PATH_PART2_DETECTIONS, exist_ok=True)
        selected_detections.to_csv((os.path.join(ic.PATH_PART2_DETECTIONS, save_filename)), index=False)
        
        return selected_detections
    
    @staticmethod
    def build_detection_windows(
    det_df: pd.DataFrame,
    window_days: int,
    ) -> pd.DataFrame:
    
        df = det_df.copy()
        df[ic.TS_COL] = pd.to_datetime(df[ic.TS_COL], errors="raise")
        
        delta = pd.Timedelta(days=window_days)
        
        df["window_start"] = df[ic.TS_COL] - delta
        df["window_end"] = df[ic.TS_COL] + delta
        
        cols = [ic.WT_ID, ic.TS_COL, "window_start", "window_end"]
        return df[cols].reset_index(drop=True)
    
    @staticmethod
    def get_top_sigs_for_detections(
    eval_df: pd.DataFrame,
    selected_detections: pd.DataFrame,
    wt_id: int,
    top_n_signals: int = 3,
) -> List[Tuple[pd.Timestamp,List[str]]]:
    
        eval_df = eval_df.copy()
        selected_detections = selected_detections.copy()
        
        eval_df[ic.TS_COL] = pd.to_datetime(eval_df[ic.TS_COL], errors="raise")
        selected_detections[ic.TS_COL] = pd.to_datetime(selected_detections[ic.TS_COL], errors="raise")
        
        selected_wt = selected_detections[selected_detections[ic.WT_ID] == wt_id].copy()
        if selected_wt.empty:
            raise ValueError(f"No detections for WT {wt_id} in selected_detections found")
        
        ts_sorted = sorted(selected_wt[ic.TS_COL].unique())
        mask = (eval_df[ic.WT_ID] ==wt_id) & (eval_df[ic.TS_COL].isin(ts_sorted))
        sub_df = eval_df.loc[mask].copy().sort_values(ic.TS_COL)
        
        if sub_df.empty:
            raise ValueError("No suited rows found in eval_df")
        
        re_cols = [c for c in sub_df.columns if c.startswith(ic.RE_PREFIX)]
        if not re_cols:
            raise ValueError("No signal columns found in sub_df.")
        
        sub_df = sub_df.set_index(ic.TS_COL)
        
        result: List[Tuple[pd.Timestamp, List[str]]] = []
        
        for ts in ts_sorted:
            if ts not in sub_df.index:
                continue
            
            row = sub_df.loc[ts]
            
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            vals = row[re_cols].to_numpy(dtype=float)
            idx_sorted = np.argsort(-vals) #descendig
            
            if top_n_signals is not None:
                idx_sorted = idx_sorted[:top_n_signals]
            
            sig_names = [re_cols[j] for j in idx_sorted]
            result.append((ts, sig_names))
            
        return result
    
    @staticmethod
    def build_normal_behavior_by_wind(
        train_pc_dir: str,
        bin_width: float = 1.0,
        wind_col: str = "Wind speed (m/s)",
        signal_cols: List[str] | None = None,
        min_samples_per_bin: int =50,
    ) -> pd.DataFrame:
        
        train_pc_dir = Path(train_pc_dir)
        files = sorted(train_pc_dir.glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files found")
        
        dfs: List[pd.DataFrame] = []
        for fp in files:
            df = pd.read_csv(fp, parse_dates=[ic.TS_COL])
            
            if ic.WT_ID not in df.columns:
                raise ValueError(f"{fp} does not contain col {ic.WT_ID}")
            if wind_col not in df.columns:
                raise ValueError(f"{fp} does not contain col {wind_col}")
            
            df[ic.TS_COL] = pd.to_datetime(df[ic.TS_COL], errors="raise")
            df = df.sort_values(ic.TS_COL)
            dfs.append(df)
        
        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        
        df_all = Part2.drop_imputations(df_all)
        
        if signal_cols is None:
            exclude = {ic.TS_COL, ic.WT_ID, wind_col, "Power (kW)"}
            num_cols = df_all.select_dtypes(include=[np.number]).columns
            signal_cols = [c for c in num_cols if c not in exclude]
        
        if not signal_cols:
            raise ValueError("No signal columns found for NBM computation")
        
        df_all = df_all.copy()
        df_all[wind_col] = df_all[wind_col].astype(float)
        df_all = df_all.dropna(subset=[wind_col, ic.WT_ID, ic.TS_COL])
        df_all["wind_bin_center"] = (
            np.floor(df_all[wind_col] / bin_width) * bin_width + bin_width/ 2.0
        )
        
        records: List[dict] = []
        # Per Signal: Grouping by WT_ID and Wind_Bin
        for sig in signal_cols:
            if sig not in df_all.columns:
                raise KeyError(f"{sig} not in df_all")
            
            df_sig = df_all[[ic.WT_ID, "wind_bin_center", sig]].copy()
            df_sig = df_sig.dropna(subset=[sig])
            
            if df_sig.empty:
                raise KeyError(f"df_sig is empty")
            
            grp = df_sig.groupby([ic.WT_ID, "wind_bin_center"], sort=False)[sig]
            
            for (wt_id, wind_center), s in grp:
                n = int(s.shape[0])
                if n == 0:
                    continue
                
                s_num = pd.to_numeric(s, errors="coerce").dropna()
                n = int(s_num.shape[0])
                
                if n < min_samples_per_bin:
                    continue

                rec = {
                    ic.WT_ID: int(wt_id),
                    NBM_COLS.SIG_NAME.value: sig,
                    NBM_COLS.WIND_BIN_CENTER.value: wind_center,
                    NBM_COLS.N_SAMPLES.value: n,
                    NBM_COLS.MEAN_VALUE.value: float(s.mean()),
                    NBM_COLS.STD_VALUE.value: float(s.std(ddof=1)),
                }
                
                records.append(rec)
        
        if not records:
            raise ValueError(f"No records created for NBM table")
        
        nbm_df = pd.DataFrame.from_records(records)
        nbm_df = nbm_df.sort_values([ic.WT_ID,
                                     NBM_COLS.SIG_NAME.value,
                                     NBM_COLS.WIND_BIN_CENTER.value])
        
            
        return nbm_df.reset_index(drop=True)
    
    @staticmethod
    def build_detection_catalog_signals(
        eval_df: pd.DataFrame,
        raw_test_df: pd.DataFrame,
        selected_detections: pd.DataFrame,
        windows_df: pd.DataFrame,
        nbm_df: pd.DataFrame,
        bin_width: float = 1.0,
        wind_col : str = "Wind speed (m/s)",
        top_n_signals: int = 10,
        train_pc_dir: Optional[str] = None,
        ) -> pd.DataFrame:
        
        eval_df = eval_df.copy()
        raw_test_df = raw_test_df.copy()
        selected_detections = selected_detections.copy()
        windows_df = windows_df.copy()
        
        eval_df[ic.TS_COL] = pd.to_datetime(eval_df[ic.TS_COL], errors="raise")
        raw_test_df[ic.TS_COL] = pd.to_datetime(raw_test_df[ic.TS_COL], errors="raise")
        selected_detections[ic.TS_COL] = pd.to_datetime(selected_detections[ic.TS_COL], errors="raise")
        windows_df[ic.TS_COL] = pd.to_datetime(windows_df[ic.TS_COL], errors="raise")
        windows_df["window_start"] = pd.to_datetime(windows_df["window_start"], errors="raise")
        windows_df["window_end"] = pd.to_datetime(windows_df["window_end"], errors="raise")
        
        eval_df = Part2.drop_imputations(eval_df)
        raw_test_df = Part2.drop_imputations(raw_test_df)
        
        if wind_col not in raw_test_df.columns:
            raise ValueError(f"Col {wind_col} is not in raw_test_df")
        
        det_with_win = selected_detections.merge(
            windows_df,
            on=[ic.WT_ID, ic.TS_COL],
            how="inner",
            validate="one_to_one"
            )
        
        if det_with_win.empty:
            raise ValueError(f"det_with_win is empty.")
        
        re_cols = [c for c in eval_df.columns if c.startswith(ic.RE_PREFIX)]
        
        if not re_cols:
            raise ValueError(f"No RE_* cols found in eval_df")
        
        eval_re_df = (
            eval_df.sort_values([ic.WT_ID, ic.TS_COL])
            .set_index([ic.WT_ID, ic.TS_COL])
        )
        
        nbm_df = nbm_df.copy()
        if ic.WT_ID not in nbm_df.columns:
            raise ValueError(f"nbm_df does not contain WT_ID column.")
        
        if NBM_COLS.SIG_NAME.value not in nbm_df.columns:
            raise ValueError(f"nbm_df does not contain col {NBM_COLS.SIG_NAME.value}.")
        
        if NBM_COLS.WIND_BIN_CENTER.value not in nbm_df.columns:
            raise ValueError(f"nbm_df does not contain col {NBM_COLS.WIND_BIN_CENTER.value}")
         
        # train_files: list[Path] = []
        # if train_pc_dir is not None:
        #     train_pc_dir = Path(train_pc_dir)
        #     train_files = sorted(train_pc_dir.glob("*.csv"))
        #     if not train_files:
        #         raise ValueError(f"No files found.")
            
        def _get_baseline_row(wt_id: int, sig_name:str, wind_center:float) -> pd.Series | None:
            nbm_sig = nbm_df[(nbm_df[ic.WT_ID] == wt_id) & 
                             (nbm_df[NBM_COLS.SIG_NAME.value] == sig_name)]
            if nbm_sig.empty:
                return None
            
            centers = nbm_sig[NBM_COLS.WIND_BIN_CENTER.value].to_numpy(dtype=float)
            mask = np.isclose(centers, float(wind_center), atol=1e-8)
            
            if not mask.any():
                return None
            
            return nbm_sig.loc[mask].iloc[0]
        
        #index event-df on wt / ts
        raw_test_df = raw_test_df.sort_values([ic.WT_ID, ic.TS_COL])
        raw_test_df.set_index([ic.WT_ID, ic.TS_COL], inplace=True)
        
        wt_ids = sorted(det_with_win[ic.WT_ID].unique().astype(int).tolist())
        #top_sigs_map: dict[int, List[Tuple[pd.Timestamp,List[str]]]] = {}
        records: List[dict] = []
        
        for wt_id in wt_ids:
            wt_id = int(wt_id)
            
            top_sigs_for_wt: List[Tuple[pd.Timestamp, List[str]]] = Part2.get_top_sigs_for_detections(
                eval_df=eval_df,
                selected_detections=selected_detections,
                wt_id = int(wt_id),
                top_n_signals=top_n_signals
            )
            
            if len(top_sigs_for_wt) ==0:
                raise KeyError(
                    f"no entries for WT_ID={wt_id}"
                )
            
            det_rows_wt = det_with_win[det_with_win[ic.WT_ID] == wt_id].copy()
            det_rows_wt = det_rows_wt.set_index(ic.TS_COL)
            
            for ts_det, re_sig_list in top_sigs_for_wt:
                if ts_det not in det_rows_wt.index:
                    raise ValueError("ts_det not in det_rows_wt")
                
                row_det = det_rows_wt.loc[ts_det]
                if isinstance(row_det, pd.DataFrame):
                    row_det = row_det.iloc[0]
                window_start = row_det["window_start"]
                window_end = row_det["window_end"]
                
                try:
                    row_ts = raw_test_df.loc[(wt_id, ts_det)]
                    if isinstance(row_ts, pd.DataFrame):
                        row_ts = row_ts.iloc[0]
                except KeyError:
                    row_ts = None
                
                wind_at_ts: float |None = None
                if row_ts is not None and wind_col in row_ts.index:
                    try:
                        wind_at_ts = float(row_ts[wind_col])
                    except (TypeError, ValueError):
                        wind_at_ts = None
                
                if wind_at_ts is not None:
                    wind_center = (
                        np.floor(wind_at_ts / bin_width) * bin_width + bin_width / 2.0
                    )
                else:
                    wind_center = np.nan
                
                wind_half_width = bin_width / 2.0
                
                for re_sig in re_sig_list:
                    if not re_sig.startswith(ic.RE_PREFIX):
                        raise ValueError(f"re_signal does not start with re_prefix: {re_sig}")
                    
                    try:
                        row_eval = eval_re_df.loc[(wt_id, ts_det)]
                        if isinstance(row_eval, pd.DataFrame):
                            row_eval = row_eval.iloc[0]
                        re_at_ts = row_eval[re_sig]
                    except (KeyError, TypeError, ValueError):
                        re_at_ts = np.nan
                    
                    
                    raw_sig = re_sig[len(ic.RE_PREFIX):]
                    
                    if raw_sig not in raw_test_df.columns:
                        raise ValueError(f"raw_sig:{raw_sig} is not in raw_test_df")
                    
                    
                    try:
                        idx_slice = pd.IndexSlice[wt_id, slice(window_start, window_end)]
                        df_event = raw_test_df.loc[idx_slice, [raw_sig, wind_col]].copy()
                    except KeyError:
                        df_event = pd.DataFrame(columns=[raw_sig, wind_col])
                    
                    df_event = df_event.dropna(subset=[raw_sig, wind_col])
                    
                    w = df_event[wind_col]
                    if isinstance(w, pd.DataFrame):
                        w = w.iloc[:, 0] # take first dupl
                    w = pd.to_numeric(w, errors="coerce")
                    
                    
                    df_event["wind_bin_center"] = (
                        np.floor(w / bin_width)* bin_width + bin_width / 2.0
                    )
                    
                    df_event = df_event.dropna(subset=["wind_bin_center"])
                    
                    n_event = int(df_event.shape[0])
                    
                    if df_event.empty or (not np.isfinite(wind_center)):
                        n_event= 0
                        mean_event = np.nan
                        std_event = np.nan
                    else:
                        mask_bin = np.isclose(
                            df_event["wind_bin_center"].to_numpy(dtype=float),
                            float(wind_center),
                        )
                        df_event_same_bin = df_event.loc[mask_bin, [raw_sig]].copy()
                        
                        x_event = df_event_same_bin[raw_sig]
                        if isinstance(x_event, pd.DataFrame):
                            x_event = x_event.iloc[:,0]
                        
                        s_event = pd.to_numeric(x_event, errors="coerce").dropna()
                        n_event = int(s_event.shape[0])
                        mean_event= float(s_event.mean()) if n_event > 0 else np.nan
                        std_event = float(s_event.std(ddof=1)) if n_event > 1 else np.nan
                        
                    if row_ts is not None and raw_sig in row_ts.index:
                        try:
                            value_at_ts = row_ts[raw_sig]
                        except (TypeError, ValueError):
                            value_at_ts = np.nan
                    else:
                        value_at_ts = np.nan
                    
                    if (wind_at_ts is not None) and np.isfinite(wind_center):
                        baseline_row = _get_baseline_row(
                            wt_id = wt_id,
                            sig_name= raw_sig,
                            wind_center=wind_center,
                        )
                    else:
                        baseline_row = None
                    
                    if baseline_row is not None:
                        n_baseline= baseline_row[NBM_COLS.N_SAMPLES.value]
                        mean_baseline= baseline_row[NBM_COLS.MEAN_VALUE.value]
                        std_baseline= baseline_row[NBM_COLS.STD_VALUE.value]
                    else:
                        n_baseline = 0
                        mean_baseline= np.nan
                        std_baseline= np.nan
                    
                        
                    if  (np.isfinite(value_at_ts) 
                        and np.isfinite(mean_baseline) 
                        and np.isfinite(std_baseline) 
                        and (std_baseline > 0.0)):
                        delta_at_ts = value_at_ts - mean_baseline
                        z_at_ts = delta_at_ts / std_baseline
                    else:
                        delta_at_ts = np.nan
                        z_at_ts = np.nan
                    
                    if (np.isfinite(mean_event) 
                        and np.isfinite(mean_baseline) 
                        and np.isfinite(std_baseline) 
                        and (std_baseline > 0.0)):
                        delta_mean = mean_event - mean_baseline
                        z_shift = delta_mean / std_baseline
                    else:
                        delta_mean = np.nan
                        z_shift= np.nan
                    
                    
                    rec = {
                        ic.WT_ID: wt_id,
                        DET_SIG_COLS.RE_AT_TS.value: re_at_ts,
                        DET_SIG_COLS.TS_DETECTION.value : ts_det,
                        DET_SIG_COLS.SIGNAL_NAME.value : raw_sig,
                        DET_SIG_COLS.WINDOW_START.value : window_start,
                        DET_SIG_COLS.WINDOW_END.value : window_end,
                        DET_SIG_COLS.WIND_CENTER.value : wind_center if np.isfinite(wind_center) else np.nan,
                        
                        DET_SIG_COLS.WIND_HALF_WIDTH.value: wind_half_width,
                        
                        DET_SIG_COLS.N_BASELINE.value: n_baseline,
                        DET_SIG_COLS.MEAN_BASELINE.value: mean_baseline,
                        DET_SIG_COLS.STD_BASELINE.value: std_baseline,
                     
                        
                        DET_SIG_COLS.N_EVENT.value: n_event,
                        DET_SIG_COLS.MEAN_EVENT.value: mean_event,
                        DET_SIG_COLS.STD_EVENT.value: std_event,
                     
                        
                        DET_SIG_COLS.VALUE_AT_TS.value: value_at_ts,
                        DET_SIG_COLS.DELTA_AT_TS.value: delta_at_ts,
                        DET_SIG_COLS.Z_AT_TS.value: z_at_ts,
                        DET_SIG_COLS.DELTA_MEAN.value: delta_mean ,
                        DET_SIG_COLS.Z_SHIFT.value: z_shift,
                    }
                    records.append(rec)
        
        if not records:
            raise ValueError("No records created.")
        
        cat_df = pd.DataFrame.from_records(records)
        cat_df = cat_df.sort_values(
            [ic.WT_ID, DET_SIG_COLS.RE_AT_TS.value],
            ascending=[True, False]
        )
        cat_df.reset_index(drop=True, inplace=True)
        
        drop_cols = [
            DET_SIG_COLS.WIND_HALF_WIDTH.value,
            DET_SIG_COLS.N_EVENT.value
        ]
        
        cat_df = cat_df.drop(
            columns=drop_cols,
            errors="raise"
        )
        
        return cat_df

    @staticmethod
    def build_detection_catalog_overview(
        selected_detections: pd.DataFrame,
        windows_df: pd.DataFrame,
        detection_catalog_signals: pd.DataFrame,
        raw_test_df: pd.DataFrame,
        df_pc: pd.DataFrame,
        wind_col: str = "Wind speed (m/s)",
        power_col: str = "Power (kW)",
        max_abs_margin_kw: float = 150.0,
        ) -> pd.DataFrame:
        
        sel_det = selected_detections.copy()
        win_df = windows_df.copy()
        sig_cat = detection_catalog_signals.copy()
        raw_df = raw_test_df.copy()
        pc_df = df_pc.copy()
        
        sel_det[ic.TS_COL] = pd.to_datetime(sel_det[ic.TS_COL], errors="raise")
        win_df[ic.TS_COL] = pd.to_datetime(win_df[ic.TS_COL], errors="raise")
        win_df["window_start"] = pd.to_datetime(win_df["window_start"], errors="raise")
        win_df["window_end"] = pd.to_datetime(win_df["window_end"], errors="raise")
        
        raw_df[ic.TS_COL] = pd.to_datetime(raw_df[ic.TS_COL], errors="raise")
        raw_df = Part2.drop_imputations(raw_df)
        
        det_with_win = sel_det.merge(
            win_df,
            on=[ic.WT_ID, ic.TS_COL],
            how="inner",
            validate="one_to_one",
        )
        
        if det_with_win.empty:
            raise ValueError(f"det_with_win is empty.")
        
        if ic.WT_ID not in sig_cat.columns:
            raise ValueError(f"detections_catalog_signals must contain WT_ID")
        if DET_SIG_COLS.TS_DETECTION.value not in sig_cat.columns:
            raise ValueError(f"detection_catalog_signals must have column {DET_SIG_COLS.TS_DETECTION.value}")
        
        sig_cat[DET_SIG_COLS.TS_DETECTION.value] = pd.to_datetime(sig_cat[DET_SIG_COLS.TS_DETECTION.value], errors="raise")
        
        sig_group = sig_cat.groupby([ic.WT_ID, DET_SIG_COLS.TS_DETECTION.value], sort=False)
        
        if "power_norm" not in pc_df.columns:
            raise ValueError(f"df_pc must contain col 'power_norm'.")
        
        pc_df = pc_df.sort_index()
        
        
        raw_df = raw_df.sort_values([ic.WT_ID, ic.TS_COL])
        
        def _expected_power_for_wind_array(wind_values: np.ndarray) -> np.ndarray:
            w = np.round(wind_values.astype(float), 2)
            exp = np.full_like(w, np.nan, dtype=float)
            
            valid = np.isin(w, pc_df.index.values)
            if not valid.any():
                return exp
            
            exp[valid] = pc_df.loc[w[valid], "power_norm"].to_numpy(dtype=float)
            return exp
        
        def _compute_pc_context_for_window(
            wt_id: int,
            window_start: pd.Timestamp,
            window_end: pd.Timestamp,
            ts_det: pd.Timestamp,
            ) -> dict:
            
            mask_wt = raw_df[ic.WT_ID].astype(int) == int(wt_id)
            mask_time = (raw_df[ic.TS_COL] >= window_start) & (raw_df[ic.TS_COL] <= window_end)
            df_win = raw_df.loc[mask_wt & mask_time, [ic.TS_COL, wind_col, power_col]].copy()
            
            df_win = df_win.dropna(subset=[ic.TS_COL, wind_col, power_col])
            n_window= int(df_win.shape[0])
            
            mask_det = mask_wt & (raw_df[ic.TS_COL] == ts_det)
            df_det_ts = raw_df.loc[mask_det, [wind_col, power_col]].copy()
            
            if not df_det_ts.empty:
                row_det = df_det_ts.iloc[0]
                wind_at_ts = row_det[wind_col]
                power_at_ts = row_det[power_col]
                exp_ts_arr = _expected_power_for_wind_array(np.array([wind_at_ts], dtype=float))
                pc_power_at_ts = exp_ts_arr[0] if np.isfinite(exp_ts_arr[0]) else np.nan
                residual_at_ts = power_at_ts - pc_power_at_ts if np.isfinite(pc_power_at_ts) else np.nan
            
            else: 
                wind_at_ts = np.nan
                power_at_ts = np.nan
                pc_power_at_ts = np.nan
                residual_at_ts = np.nan
                
            if n_window > 0:
                wind_arr = df_win[wind_col].to_numpy(dtype=float)
                power_arr = df_win[power_col].to_numpy(dtype=float)
                exp_arr = _expected_power_for_wind_array(wind_arr)
                
                res = power_arr - exp_arr
                
                finite_mask = np.isfinite(exp_arr) & np.isfinite(power_arr)
                res = res[finite_mask]
                
                if res.size > 0:
                    within  = np.abs(res) <= max_abs_margin_kw
                    below = res < -max_abs_margin_kw
                    above = res > max_abs_margin_kw
                    
                    share_within = float(within.mean())
                    share_below = float(below.mean())
                    share_above = float(above.mean())
                    
                else:
                    share_within = np.nan
                    share_below = np.nan
                    share_above = np.nan
            
            else:
                share_within = np.nan
                share_below = np.nan
                share_above = np.nan
            
            return {
                DET_OVERVIEW_COLS.WIND_AT_TS.value: wind_at_ts,
                DET_OVERVIEW_COLS.POWER_AT_TS.value: power_at_ts,
                DET_OVERVIEW_COLS.PC_POWER_AT_TS.value: pc_power_at_ts ,
                DET_OVERVIEW_COLS.RESIDUAL_AT_TS.value: residual_at_ts,
                DET_OVERVIEW_COLS.SHARE_WITHIN_PC_BAND_WINDOW.value: share_within,
                DET_OVERVIEW_COLS.SHARE_BELOW_PC_BAND_WINDOW.value: share_below,
                DET_OVERVIEW_COLS.SHARE_ABOVE_PC_BAND_WINDOW.value: share_above,
            }
        
        def _compute_pitch_asymmetrie_for_detection(wt_id:int, ts_det:pd.Timestamp, eps: float = 1e-3,) -> dict:
            
            mask = (
                (raw_df[ic.WT_ID].astype(int) == int(wt_id)) & (raw_df[ic.TS_COL] == ts_det)
            )
            
            df_det_ts = raw_df.loc[mask].copy()
            if df_det_ts.empty:
                return {DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value:False}
            row = df_det_ts.iloc[0]
            
            pitch_cols = [c for c in row.index if "pitch" in c.lower()]
            pitch_vals = row[pitch_cols].astype(float).to_numpy()
            
            if pitch_vals.size < 2:
                raise ValueError(f"Not all pitch values extracted")
            
            
            asym = np.max(pitch_vals) - np.min(pitch_vals)
            tag_asym = bool(asym > eps) # eps = noise
            
            return {
                DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value: tag_asym,
            }
        
        def _aggregate_signal_tags_for_detection(wt_id: int, ts_det: pd.Timestamp, eps:float = 1e-3) -> dict:
            try: 
                grp = sig_group.get_group((wt_id, ts_det))
            except KeyError:
                return {
                    DET_OVERVIEW_COLS.N_TOP_SIGNALS.value: 0,
                    DET_OVERVIEW_COLS.TAG_TEMP_HIGH.value: False,
                    DET_OVERVIEW_COLS.TAG_VIBRATION_HIGH.value: False ,
                    DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value: False ,
                }
            
            n_top_signals = int(grp.shape[0])

            sig_names = grp[DET_SIG_COLS.SIGNAL_NAME.value].astype(str).tolist()
            
            def _is_temp(name:str) -> bool:
                lower = name.lower()
                return ("temp" in lower) or ("°c" in lower) or ("temperature" in lower)
            
            def _is_vibration(name: str) -> bool:
                lower =name.lower()
                return ("vibration" in lower) or ("acceler" in lower)


            
            pitch = _compute_pitch_asymmetrie_for_detection(wt_id, ts_det, eps= eps)
            pitch_asym = pitch[DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value]
            
            temp_flag = any(_is_temp(n) for n in sig_names)
            vib_flag = any(_is_vibration(n) for n in sig_names)
            pitch_flag = pitch_asym
            
            return {
                DET_OVERVIEW_COLS.N_TOP_SIGNALS.value: n_top_signals,
                DET_OVERVIEW_COLS.TAG_TEMP_HIGH.value: temp_flag,
                DET_OVERVIEW_COLS.TAG_VIBRATION_HIGH.value: vib_flag,
                DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value: pitch_flag,
            }

        
        records: List[dict] = []
        
        for _, row in det_with_win.iterrows():
            wt_id = int(row[ic.WT_ID])
            ts_det = row[ic.TS_COL]
            window_start = row["window_start"]
            window_end = row["window_end"]
            re_mean = float(row[ic.MEAN_LOSS_PER_SAMPLE])
            
            pc_ctx = _compute_pc_context_for_window(
                wt_id = wt_id,
                window_start= window_start,
                window_end=window_end,
                ts_det=ts_det,
            )
            
            sig_info = _aggregate_signal_tags_for_detection(
                wt_id = wt_id,
                ts_det=ts_det,
            )
            
            rec = {
                ic.WT_ID: wt_id,
                DET_OVERVIEW_COLS.TS_DETECTION.value: ts_det,
                DET_OVERVIEW_COLS.WINDOW_START.value: window_start,
                DET_OVERVIEW_COLS.WINDOW_END.value: window_end,
                DET_OVERVIEW_COLS.RE_MEAN.value: re_mean,
                DET_OVERVIEW_COLS.WIND_AT_TS.value: pc_ctx[DET_OVERVIEW_COLS.WIND_AT_TS.value],
                DET_OVERVIEW_COLS.POWER_AT_TS.value: pc_ctx[DET_OVERVIEW_COLS.POWER_AT_TS.value],
                DET_OVERVIEW_COLS.PC_POWER_AT_TS.value: pc_ctx[DET_OVERVIEW_COLS.PC_POWER_AT_TS.value],
                DET_OVERVIEW_COLS.RESIDUAL_AT_TS.value: pc_ctx[DET_OVERVIEW_COLS.RESIDUAL_AT_TS.value] ,
                DET_OVERVIEW_COLS.SHARE_WITHIN_PC_BAND_WINDOW.value: pc_ctx[DET_OVERVIEW_COLS.SHARE_WITHIN_PC_BAND_WINDOW.value],
                DET_OVERVIEW_COLS.SHARE_BELOW_PC_BAND_WINDOW.value: pc_ctx[DET_OVERVIEW_COLS.SHARE_BELOW_PC_BAND_WINDOW.value],
                DET_OVERVIEW_COLS.SHARE_ABOVE_PC_BAND_WINDOW.value: pc_ctx[DET_OVERVIEW_COLS.SHARE_ABOVE_PC_BAND_WINDOW.value] ,
                DET_OVERVIEW_COLS.N_TOP_SIGNALS.value: sig_info[DET_OVERVIEW_COLS.N_TOP_SIGNALS.value],
                DET_OVERVIEW_COLS.TAG_TEMP_HIGH.value: sig_info[DET_OVERVIEW_COLS.TAG_TEMP_HIGH.value],
                DET_OVERVIEW_COLS.TAG_VIBRATION_HIGH.value: sig_info[DET_OVERVIEW_COLS.TAG_VIBRATION_HIGH.value],
                DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value: sig_info[DET_OVERVIEW_COLS.TAG_PITCH_ASYMMETRY.value],
            }
            
            records.append(rec)
        
        if not records:
            raise ValueError("No records created for detection_catalog_overwiew")
        
        overview_df = pd.DataFrame.from_records(records)
        overview_df = overview_df.sort_values(
            [DET_OVERVIEW_COLS.RE_MEAN.value],
            ascending=[False]
        ).reset_index(drop=True)
        
        drop_cols = [
            DET_OVERVIEW_COLS.N_TOP_SIGNALS.value,
            DET_OVERVIEW_COLS.WINDOW_START.value,
            DET_OVERVIEW_COLS.WINDOW_END.value,
        ]
        
        overview_df = overview_df.drop(
            columns=drop_cols,
            errors="raise"
        )
        
        return overview_df
    
    
    @staticmethod
    def infer_cut_in_from_pc(
        df_pc: pd.DataFrame,
        col: str = "power_norm",
        rel_threshold: float = 0.00,
        min_consecutive: int = 5,
    ) -> float:
        #TODO
        # too complicated - better approch select first item that > 0.
        if col not in df_pc.columns:
            raise ValueError(f"Column '{col}' not found in df_pc")
        
        wind = df_pc.index.to_numpy(dtype=float)
        power = df_pc[col].to_numpy(dtype=float)
        
        if power.size == 0:
            raise ValueError("df_pc is empty")
        
        p_max = np.nanmax(power)
        
        if not np.isfinite(p_max) or p_max <= 0.0:
            raise ValueError("Invalid max power in df_pc")
        
        thr = rel_threshold * p_max
        
        mask = power >= thr
        
        run_length = 0
        for i, ok in enumerate(mask):
            if ok:
                run_length += 1
                if run_length >= min_consecutive:
                    idx_start = i - run_length +1
                    return float(wind[idx_start])
                
            else:
                run_length = 0
        raise ValueError("Could not determine cut-in from power curve")
    
    @staticmethod
    def compute_sliding_window_scores(
        eval_df: pd.DataFrame,
        window_length_units: int,
        wt_col: str = ic.WT_ID,
        ts_col: str = ic.TS_COL,
        loss_col: str = ic.MEAN_LOSS_PER_SAMPLE,
        new_col: str = "window_mean_loss",
        ) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: [wt_col, ts_col, new_col], each row corresponds to the center of a sliding window of length window_length_units and new_col is the mean loss of that window belonging to ts_col.
        """
        if window_length_units <= 0:
            raise ValueError(f"window_length_units must be > 0")
        
        df = eval_df.copy()
        
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
        df = df.sort_values([wt_col, ts_col])
        
        
        
        rolling_mean = (
            df.groupby(wt_col)[loss_col]
            .rolling(
                window=window_length_units,
                min_periods=window_length_units,
                center=True
            )
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        df[new_col] = rolling_mean
        df_valid = df.dropna(subset=[new_col]).copy()
        
        result = df_valid[[wt_col, ts_col, new_col]].reset_index(drop=True)
        
        return result
    
    @staticmethod
    def select_top_k_windows_per_turbine(
        window_scores_df: pd.DataFrame,
        top_k: int,
        wt_col: str = ic.WT_ID,
        ts_col: str = ic.TS_COL,
        score_col: str = "window_mean_loss",
        ) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: This dataframe has the same cols as window_scores_df from compute_sliding_window_scores().
        """
        df = window_scores_df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
        df = df.sort_values([wt_col, score_col, ts_col], ascending=[True,False,True])
        
        if top_k is None:
            df = df.reset_index(drop=True)    
            return df
        
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0")
        
        grouped = df.groupby(wt_col, group_keys=False)
        top_k_df = grouped.head(top_k).reset_index(drop=True)
        
        return top_k_df
        
    @staticmethod
    def build_event_windows_around_topk(
        top_k_df: pd.DataFrame,
        event_half_width_units: int,
        wt_col: str = ic.WT_ID,
        ts_col: str = ic.TS_COL,
        window_start_col: str = "window_start",
        window_end_col: str ="window_end",
        ) -> pd.DataFrame:
        
        if event_half_width_units <= 0:
            raise ValueError(f"event_half_width_units must be > 0")
        
        df = top_k_df.copy()
        
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
        
        half_delta = pd.to_timedelta(event_half_width_units *10, unit="m")
        
        df[window_start_col] = df[ts_col] - half_delta
        df[window_end_col] = df[ts_col] + half_delta
        df = df.sort_values([wt_col, ts_col]).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def merge_event_windows(
        events_df: pd.DataFrame,
        wt_col: str = ic.WT_ID,
        window_start_col: str = "window_start",
        window_end_col: str = "window_end",
        min_overlap_ratio: float= 0.5,
        ) -> pd.DataFrame:
        
        """Heuristic to merge top-k ts if they overlap with min_overlap_ratio.
            events_df is from build_event_windows_around_topk().
        Returns:
            pd.DataFrame: [wt_col, window_start_col, window_end_col] - contains events of top-k ts by scoring.
        """
        
        if not (0.0 < min_overlap_ratio < 1.0):
            raise ValueError(f"min_overlap_ratio must be in (0,1)")
        
        if events_df.empty:
            raise ValueError(f"events_df is empty!")
        
        df = events_df.copy()
        
        df[window_start_col] = pd.to_datetime(df[window_start_col], errors="raise")
        df[window_end_col] = pd.to_datetime(df[window_end_col], errors="raise")
        
        df = df.sort_values([wt_col, window_start_col, window_end_col]).reset_index(drop=True)
        
        wt_vals = df[wt_col].to_numpy()
        start_vals =df[window_start_col].to_numpy()
        end_vals = df[window_end_col].to_numpy()
        
        merged_records: List[dict] = []
        
        current_wt = None
        current_start = None
        current_end = None
        
        for i in range(len(df)):
            wt = wt_vals[i]
            start_next = start_vals[i]
            end_next = end_vals[i]
            
            #new wt
            if (current_wt is None) or (wt != current_wt):
                if current_wt is not None:
                    merged_records.append(
                        {
                            wt_col: current_wt,
                            window_start_col: current_start,
                            window_end_col: current_end,
                        }
                    )
                
                current_wt = wt
                current_start = start_next
                current_end = end_next
                
                continue
            
            # wt not changed
            start_inters = max(current_start, start_next)
            end_inters = min(current_end, end_next)
            
            if end_inters > start_inters:
                #intersection
                len_inters = end_inters - start_inters
                len_current = current_end - current_start
                len_next = end_next - start_next
                len_short = min(len_current, len_next)
                
                if len_short <= pd.Timedelta(0):
                    overlap_ratio = 0.0
                
                else:
                    overlap_ratio = len_inters / len_short
            
            else:
                overlap_ratio = 0.0
            
            if overlap_ratio >= min_overlap_ratio:
                # do merge
                if start_next < current_start:
                    current_start = start_next
                
                if end_next > current_end:
                    current_end = end_next
            
            else:
                #close event
                merged_records.append(
                    {
                        wt_col: current_wt,
                        window_start_col: current_start,
                        window_end_col: current_end,
                    }
                )
                current_start = start_next
                current_end = end_next
        
        #last event
        if current_wt is not None:
            merged_records.append(
                {
                    wt_col: current_wt,
                    window_start_col: current_start,
                    window_end_col: current_end,
                }
            )
        
        merged_df = pd.DataFrame.from_records(merged_records)
        
        if merged_df.empty:
            raise ValueError(f"merged_df is empty")
        
        merged_df = merged_df.sort_values([wt_col, window_start_col, window_end_col]).reset_index(drop=True)
        
        return merged_df
    
    @staticmethod
    def build_event_detections_from_windows(
        eval_df: pd.DataFrame,
        merged_windows_df: pd.DataFrame,
        wt_col: str = ic.WT_ID,
        ts_col: str = ic.TS_COL,
        loss_col: str = ic.MEAN_LOSS_PER_SAMPLE,
        window_start_col: str = "window_start",
        window_end_col: str = "window_end",
        mean_col: str = "event_mean_loss",
        max_col: str = "event_max_loss",
        ) -> pd.DataFrame:
        
        if merged_windows_df.empty:
            return merged_windows_df[[wt_col, window_start_col, window_end_col]].assign(
                **{ts_col: pd.NaT, loss_col: np.nan,
                mean_col: np.nan,
                max_col: np.nan}) [[wt_col, ts_col, loss_col,mean_col, max_col, window_start_col, window_end_col]].iloc[0:0].copy()
        
        df_eval = eval_df.copy()
        df_eval[ts_col] = pd.to_datetime(df_eval[ts_col], errors="raise")
        
        df_wins = merged_windows_df.copy()
        df_wins[window_start_col] = pd.to_datetime(df_wins[window_start_col], errors="raise")
        df_wins[window_end_col] = pd.to_datetime(df_wins[window_end_col], errors="raise")
        
        df_eval = df_eval.sort_values([wt_col, ts_col])
        df_wins = df_wins.sort_values([wt_col, window_start_col, window_end_col])
        
        records: List[dict] = []
        
        for wt_id, win_grp in df_wins.groupby(wt_col, sort=False):
            mask_wt = df_eval[wt_col] == wt_id
            df_eval_wt = df_eval.loc[mask_wt, [ts_col, loss_col]].copy()
            
            if df_eval_wt.empty:
                continue
            
            ts_vals = df_eval_wt[ts_col].to_numpy()
            loss_vals = df_eval_wt[loss_col].to_numpy()
            
            for _, row in win_grp.iterrows():
                w_start = row[window_start_col]
                w_end = row[window_end_col]
                
                in_window = (ts_vals >= w_start) & (ts_vals <= w_end)
                
                if not in_window.any():
                    continue
                
                vals = loss_vals[in_window]
                idx_local = np.argmax(vals)
                idx_global = np.nonzero(in_window)[0][idx_local]
                
                ts_det = ts_vals[idx_global]
                loss_det = float(loss_vals[idx_global])
                
                rec = {
                    wt_col: wt_id,
                    ts_col: ts_det,
                    loss_col: loss_det,
                    mean_col: float(vals.mean()),
                    max_col: float(vals.max()),
                    window_start_col:w_start,
                    window_end_col: w_end,
                }
                records.append(rec)
        
        if not records:
            return pd.DataFrame(columns=[wt_col, ts_col, loss_col, mean_col, max_col, window_start_col, window_end_col])
        
        det_df = pd.DataFrame.from_records(records)
        
        det_df = det_df.sort_values([wt_col, ts_col]).reset_index(drop=True)
        
        return det_df
    
    @staticmethod
    def build_event_detections_sliding_windows_pipeline(
        eval_df: pd.DataFrame,
        window_length_units: int,
        top_k: int,
        event_half_width_units:int,
        wt_col: str = ic.WT_ID,
        ts_col: str = ic.TS_COL,
        loss_col: str = ic.MEAN_LOSS_PER_SAMPLE,
        window_score_col: str = "window_mean_loss",
        window_start_col: str = "window_start",
        window_end_col: str = "window_end",
        min_overlap_ratio: float = 0.5,
        mean_col: str = "event_mean_loss",
        max_col: str = "event_max_loss",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ 1. Per Turbine it calculates mean over a sliding window. \n
            2. Top-k timestamps will be selected \n
            3. build windows around these timestamps \n
            4. overlapping windows with at least min_overlap_ratio will be merged to an event \n
            5. within this event retrieve max. RE and use this corresponding timestamp as a detection
        Args:
            eval_df (pd.DataFrame): Contains the RE losses
            window_length_units (int): 1 => 1 time step => e.g. 10 min
            top_k (int): Number of top k windows per wind turbine from sliding_windows-scores
            event_half_width_units (int): Half window width around every top-k-timestamp
            wt_col (str, optional): Defaults to ic.WT_ID.
            ts_col (str, optional):  Defaults to ic.TS_COL.
            loss_col (str, optional): . Defaults to ic.MEAN_LOSS_PER_SAMPLE.
            window_score_col (str, optional): Defaults to "window_mean_loss".
            window_start_col (str, optional): Defaults to "window_start".
            window_end_col (str, optional): Defaults to "window_end".
            min_overlap_ratio (float, optional): Ratio to merge overlapping windows. Defaults to 0.5.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: [[wt_col, ts_col, loss_col, RE_* ], [wt_col, ts_col, window_start_col, window_end_col]]
        """
        window_scores = Part2.compute_sliding_window_scores(
            eval_df= eval_df,
            window_length_units=window_length_units,
            wt_col=wt_col,
            ts_col=ts_col,
            loss_col=loss_col,
            new_col=window_score_col,
        )
        
        topk_df = Part2.select_top_k_windows_per_turbine(
            window_scores_df=window_scores,
            top_k=top_k,
            wt_col=wt_col,
            ts_col=ts_col,
            score_col=window_score_col,
        )
        
        window_scores= None
        
        events_raw = Part2.build_event_windows_around_topk(
            top_k_df=topk_df,
            event_half_width_units=event_half_width_units,
            wt_col=wt_col,
            ts_col=ts_col,
            window_start_col=window_start_col,
            window_end_col=window_end_col,
        )
        topk_df = None
        
        events_merged = Part2.merge_event_windows(
            events_df=events_raw,
            wt_col=wt_col,
            window_start_col=window_start_col,
            window_end_col=window_end_col,
            min_overlap_ratio=min_overlap_ratio,
        )
        
        events_raw = None
        
        event_det = Part2.build_event_detections_from_windows(
            eval_df = eval_df,
            merged_windows_df=events_merged,
            wt_col=wt_col,
            ts_col=ts_col,
            loss_col=loss_col,
            window_start_col=window_start_col,
            window_end_col=window_end_col,
            mean_col=mean_col,
            max_col=max_col,
        )
        
        keys = event_det[[wt_col, ts_col, mean_col, max_col]].copy()
        
        selected_detections = keys.merge(
            eval_df,
            on=[wt_col, ts_col],
            how="left",
            validate="1:1",
        ).sort_values([wt_col, ts_col]).reset_index(drop=True)
        
        windows_df = (
            event_det[[wt_col, ts_col, mean_col, max_col, window_start_col, window_end_col]]
            .sort_values([wt_col, ts_col])
            .reset_index(drop=True)
        )
        
        return selected_detections, windows_df
    
    
class ks_test:
    Agg = Literal["mean", "median"]

    @classmethod
    def split_pre_post(
        cls,
        df: pd.DataFrame,
        col: str,
        ts_det: pd.Timestamp,
        offset: pd.Timedelta,
        ts_col: str = ic.TS_COL,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: [ts_det - offset, ts_det), [ts_det+delta, ts_det + offset + delta)
        """
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col)
        
        ts = df[ts_col].to_numpy()
        vals = df[col].to_numpy(dtype=float)
        
        tm = np.datetime64(ts_det - offset)
        t = np.datetime64(ts_det)
        tp = np.datetime64(ts_det + offset)
        dt = np.timedelta64(10, "m")
        pre = vals[(ts >= tm) & (ts < t)]
        post = vals[(ts >= t+dt) & (ts < tp + dt)]
        
        return pre,post

    @classmethod
    def acf(cls,x: np.ndarray, k:int) -> np.ndarray:
        """ Computes auto correlation using the biased estimator
        Returns:
            np.ndarray: if x is constant NaNs will be returned.
        """
        
        if k <= 0:
            return np.array([], dtype=float)
        
        k = min(k, x.size - 1)
        x0 = x - np.mean(x)
        denom = float(np.dot(x0,x0))
        if denom == 0.0:
            return np.full(k, np.nan, dtype=float)
        
        results = np.empty(k, dtype=float)
        
        for h in range(1,k +1):
            # sum^(n-1)_(i=h) x0(i)*x0(i-h) -> reindexing: sum^(n-h-1)_(i=0) x0(i+h) x0(i) 
            results[h-1] = float(np.dot(x0[h:], x0[:-h]) / denom)
            
        return results 

    @classmethod
    def cliffs_delta(cls, pre: np.ndarray, post: np.ndarray) -> float:
        """Computes cliffs delta: P(post > pre) - P(post < pre), delta in [-1,1]
        """
        pre = np.asarray(pre, dtype=float)
        post = np.asarray(post, dtype=float)
        
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        
        if post.size == 0 or pre.size == 0:
            raise ValueError(f"pre or post cannot be empty: pre.size = {pre.size}, post.size = {post.size}")
        
        
        greater = 0
        less = 0
        for i in post:
            greater += int(np.sum(i > pre))
            less += int(np.sum(i < pre))
        
        return float((greater - less) / (pre.size * post.size))

    @classmethod
    def _acf_abs_max(cls, acf: np.ndarray) -> float:
        acf = np.asarray(acf, dtype=float)
        if acf.size == 0:
            raise ValueError(f"acf is empty")
        
        acf_abs = np.abs(acf)
        return np.max(acf_abs)
    
        # cls,
        # eval_blocks: pd.DataFrame,
        # ) -> int:
        # candidates = eval_blocks[eval_blocks["meets_cond"] == True]
        # if len(candidates) > 0:
        #     blk_size = int(candidates["block_size"].min())
        #     return blk_size
        # else: 
        #     raise ValueError(f"No blocks fullfilled the condition.")
    #####################
    # new approach
    #####################
    @dataclass(frozen=True)
    class ADFResult:
        test_stat: float
        p_value: float
        lag: int
        n_obs: int
        
    @classmethod
    def adf_test(cls,
                 x: np.ndarray,
                 alpha: float = 0.05,
                 regression: Literal["c", "ct", "ctt", "n"] = "c",
                 autolag: Literal["AIC", "BIC", "t-stat", None] = "AIC",
                 ) -> Tuple["ks_test.ADFResult", bool]:
        """
        Returns:
            (result, is_stationary): is_stationary == (p_value <= alpha)
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            raise ValueError(f"ADF needs more samples. n={x.size}")
        
        adf = adfuller(x, regression=regression, autolag=autolag)
        res = cls.ADFResult(
            test_stat=float(adf[0]),
            p_value=float(adf[1]),
            lag=int(adf[2]),
            n_obs=int(adf[3]),
        )
        return res, bool(res.p_value <= float(alpha))
    
    @classmethod
    def difference_until_stationary(cls,
                                x: np.ndarray,
                                alpha: float = 0.05,
                                regression: Literal["c", "ct", "ctt", "n"] = "c",
                                autolag: Literal["AIC", "BIC", "t-stat", None] = "AIC",
                                max_d: int = 24,
                                ) -> Dict[str, Any]:
        """
        Checks stationarity, if not apply differencing operator up to max_d.
        Returns:
            Dict[str, Any]: "x","d","adf_pre","stationary_pre","adf_post","stationary_post","adf_path","stationary_path"
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        
        x_tmp = x
        d_selected = 0
        adf_path: List[ks_test.ADFResult] = []
        stationary_path: List[bool] = []
        for d_tmp in range(0, int(max_d) + 1):
            adf_res, is_stationary = cls.adf_test(x_tmp, alpha=alpha, regression=regression, autolag=autolag)
            
            adf_path.append(adf_res)
            stationary_path.append(bool(is_stationary))
            
            if is_stationary:
                d_selected = d_tmp
                break
            if d_tmp == int(max_d):
                d_selected = int(d_tmp)
                break
            
            x_tmp = np.diff(x_tmp, n=1)
            if x_tmp.size < 20:
                raise ValueError(f" Series became too short. d={d_tmp}, n={x_tmp.size}")
        
        adf_pre = adf_path[0]
        stat_pre = stationary_path[0]
        adf_post = adf_path[-1]
        stat_post = stationary_path[-1]
        
        return {
            "x": x_tmp,
            "d": int(d_selected),
            "adf_pre": adf_pre,
            "stationary_pre": bool(stat_pre),
            "adf_post": adf_post,
            "stationary_post": bool(stat_post),
            "adf_path": adf_path,
            "stationary_path": stationary_path,
        }
            
    
    @staticmethod
    def _aicc_from_aic(aic: float, n: int, k: int) -> float:
        """
        AICc = AIC + 2k(k+1)/(n-k-1)
        Returns NaN if undefined
        """
        
        if (n is None) or (k is None) or (n-k-1) <=0:
            return float("nan")
        return float(aic + (2.0 * k * (k+1)) / (n-k-1))
    
    @classmethod
    def select_ar_order_aic(
        cls,
        x: np.ndarray,
        p_max: int = 24,
        include_const: bool=True,
    )-> pd.DataFrame:
        """
        Fit ARIMA (p,0,0) for p=0...p_max
        p=0 => ARIMA(0,0,0): white noise around mean if include_const=True.
        x is cleaned to finite values.
        Returns:
            pd.DataFrame: prefer, aic, aicc, n_over_k_lt_40, p, n , k, 
        """
        
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            raise ValueError(f"x too small: n={x.size}")
        
        trend: Literal["c", "n"] = "c" if include_const else "n"
        rows: List[Dict[str,Any]] = []
        for p in range(0, int(p_max) + 1):
            
            try:
                model = AutoReg(x, lags=int(p), trend=trend, old_names=False)
                fit = model.fit()
                
                aic = float(fit.aic)
                aicc = float(fit.aicc)
                
                n = int(fit.nobs)
                k = int(fit.params.size)
                
                aicc = cls._aicc_from_aic(aic, n=n, k=k)
                n_over_k = (n/k) if k > 0 else float("inf")
                n_over_k_lt_40 = n_over_k < 40.0
                
                rows.append(
                    {
                        "p": int(p),
                        "n": n,
                        "k": k,
                        "n_over_k_lt_40": n_over_k_lt_40,
                        "aic": aic,
                        "aicc": aicc,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "p": int(p),
                        "n": int(x.size),
                        "k": np.nan,
                        "n_over_k_lt_40": np.nan,
                        "aic": np.nan,
                        "aicc": np.nan,
                    }
                )
        df = pd.DataFrame(rows)
        
        def _prefer(row: pd.Series) -> str:
            if bool(row.get("n_over_k_lt_40", False)) and np.isfinite(row.get("aicc", np.nan)):
                return "aicc"
            return "aic"
        
        df["prefer"] = df.apply(_prefer, axis=1)
        
        df = df.sort_values(["aicc", "aic", "p"], ascending=[True, True, True]).reset_index(drop=True)
        return df
        
    @classmethod
    def _pick_best_p(cls,
                     table: pd.DataFrame,
                     ) -> int:
        """Choose p according to prefer rule:
            if AICc is prefered, pick the smallest.
            if AIC  ..., pick the smallest.
        Returns:
            int: p
        """
        if "prefer" not in table.columns:
            raise ValueError("table does not contain the column 'prefer'")
        
        df = table.copy()
        df = df[np.isfinite(df["aic"].to_numpy(dtype=float))].copy()
        if df.empty:
            raise ValueError("No successful ARIMA (AR) fits in order_table.")
        
        df_aicc = df[(df["prefer"] =="aicc") & np.isfinite(df["aicc"].to_numpy(dtype=float))]
        if not df_aicc.empty:
            best = df_aicc.sort_values(["aicc","p"], ascending=[True, True]).iloc[0]
            return int(best["p"])
        
        best = df.sort_values(["aic", "p"], ascending=[True,True]).iloc[0]
        return int(best["p"])
    
    @classmethod
    def fit_arima_ar_only(cls,
                          x: np.ndarray,
                          p: int,
                          include_const: bool = True,
                          ):
        
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            raise ValueError(f"Not enough samples. n={x.size}")
        trend: Literal["c", "n"] = "c" if include_const else "n"
        results = ARIMA(x, order=(int(p), 0, 0), trend=trend).fit()
        return results
    
    @dataclass(frozen=True)
    class PreWhitenResult:
        p: int
        phi: np.ndarray
        intercept: float
        resid_pre: np.ndarray
        resid_post: np.ndarray
    
    @classmethod
    def prewhitening_residuals(cls,
                               pre: np.ndarray,
                               post: np.ndarray,
                               p: int,
                               include_const: bool = True,
                               ) -> PreWhitenResult:
        if pre.size <= p:
            raise ValueError(f"pre too short. len(pre)= {pre.size}")
        if post.size == 0:
            raise ValueError("post empty")
        
        trend = "c" if include_const else "n"
        model = AutoReg(pre, lags=p, trend=trend, old_names=False)
        res = model.fit()
        
        params = np.asarray(res.params, dtype=float)
        if include_const:
            intercept = float(params[0])
            phi = params[1:]
        else:
            intercept= 0.0
            phi = params
            
        resid_pre = np.asarray(res.resid, dtype=float)
        
        history = np.concatenate([pre[-p:], post], axis=0)
        resid_post = np.empty(post.size, dtype=float)
        
        for t in range(post.size):
            idx = t + p
            y_hat = intercept
            #y_hat += sum_{i=1...p} phi_i *y_{t_i}
            # y_{t_i} is history[idx - i]
            for i in range(1, p+1):
                y_hat += phi[i-1] * history[idx - i]
            resid_post[t] = history[idx] - y_hat
        
        return cls.PreWhitenResult(
            p=p,
            phi=np.asarray(phi,dtype=float),
            intercept=intercept,
            resid_pre=resid_pre,
            resid_post= resid_post,
        )
    
    @classmethod
    def standardize_by_pre_sigma(cls,
                                 resid_pre: np.ndarray,
                                 resid_post: np.ndarray,
                                 ddof: int = 1,
                                 ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Standardize both with sigma_pre
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: pre, post, sigma_pre
        """
        
        pre = np.asarray(resid_pre, dtype=float)
        post = np.asarray(resid_post, dtype=float)
        
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        
        if pre.size < 5 or post.size < 5:
            raise ValueError(f"Too few residual samples to standardize")
        
        sigma_pre = float(np.std(pre, ddof=int(ddof)))
        if not np.isfinite(sigma_pre) or sigma_pre <= 0.0:
            raise ValueError(f"sigma_pre invalid: {sigma_pre}")
        
        return pre / sigma_pre, post/ sigma_pre, sigma_pre
    
    @staticmethod
    def check_white_noise(residuals: np.ndarray, lags:int=24, alpha:float=0.05) -> Tuple[bool, float, float, int, float]:
        """ ljung-box test
            White noise Test on residuals

        Returns:
            Tuple[bool, float, int, float]: (is_white_noise, p_val, alpha, lags, test_stat)
        """
        results = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        p_val = results["lb_pvalue"].iloc[0]
        test_stat = results["lb_stat"].iloc[0]
        
        reject_h0 = p_val <= alpha
        return reject_h0, p_val, alpha, lags, test_stat
    
    @classmethod
    def ks_on_residuals(cls,
                        pre: np.ndarray,
                        post: np.ndarray,
                        method: Literal["auto", "exact", "asymp"] = "exact",
                        alternative:str = "two-sided",
                        ) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: "Ks_D","ks_p","n_pre","n_post","alternative","method"
        """
        pre = np.asarray(pre, dtype=float)
        post = np.asarray(post, dtype=float)
        
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        
        if pre.size ==0 or post.size ==0:
            raise ValueError("Residual arrays must be non empty.")
        
        ks = ks_2samp(pre, post, alternative=alternative, method=method)
        return {
            "ks_D": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "n_pre": int(pre.size),
            "n_post": int(post.size),
            "alternative": alternative,
            "method": method,
        }
    
    
    @classmethod
    def plot_residual_diagnostics(cls,
        resid: np.ndarray,
        lags: int = 24,
        title: str ="ACF on Residuals - White Noise Check",
        save_path: Path = Path(ic.PATH_PRINTS) / "resid_diags.png",
        dpi: int = 300,
    ):
        fig, ax = plt.subplots(figsize=(10,5))
        plot_acf(resid, lags=lags, ax=ax, title=title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        mh = Line2D([0], [0], marker="o", color="C0", linestyle="", label="Autocorrelation" )
        ih = mpatches.Patch(color="C0", alpha=0.2, label="Approx. 95% significance bounds (H0: white noise)")
        ax.legend(handles=[mh, ih], loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.show()
    
    @staticmethod
    def _ecdf(x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        xs = np.sort(x)
        n = xs.size
        ys = np.arange(1, n+1, dtype=float) / float(n)
        return xs, ys
    
    @classmethod
    def plot_ecdf(cls,
                  pre: np.ndarray,
                  post:np.ndarray,
                  title: str= "ECDF pre vs post",
                  save_path:Path =  Path(ic.PATH_PRINTS) / "ecdf.png",
                  dpi: int = 300
                  ):
        
        pre = ECDF(pre)
        post = ECDF(post)
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.step(pre.x, pre.y, label="Pre", where="post")
        ax.step(post.x, post.y, label="Post", where="post")
        
        ax.set_xlabel("Standardized Residuals")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout
        fig.savefig(save_path, dpi=dpi)
        plt.show()
    
        
    @classmethod
    def run_prewhitening_ks_pipeline(cls,
        df: pd.DataFrame,
        col: str,
        ts_det: pd.Timestamp,
        offset: pd.Timedelta,
        ts_col: str = ic.TS_COL,
        adf_alpha: float = 0.05,
        adf_regression: Literal["c", "ct", "ctt", "n"] = "c",
        adf_autolag: Literal["AIC", "BIC", "t-stat", None] = "AIC",
        max_d: int = 5,
        p_max: int = 24, # AR order
        include_const: bool = True,
        ks_method: Literal["auto", "exact", "asymp"] = "exact",
        ks_alternative: str = "two-sided",
        acf_k: int = 24, #max lags,
    ) -> Dict[str, Any]:
        """ - split pre/post
            - stationary check + differencing (de-trending)
            - AR order selection on pre
            - fit ARIMA(p,0,0) (AR) on pre, then apply on post (decorrelation)
            - residuals
            - standardization (on pre, then apply on post)
            - KS test
            - ACF diag. on residuals (pre,post)

        Args:
            df (pd.DataFrame): signals of wt_i
            col (str): signal col to inspect
            ts_det (pd.Timestamp): detection time
            offset (pd.Timedelta): Used to construct windows before and after ts_det
            ts_col (str, optional): Defaults to ic.TS_COL.
            adf_alpha (float, optional): Significance level for ADF-Tets. Defaults to 0.05.
            adf_regression (Literal[&quot;c&quot;, &quot;ct&quot;, &quot;ctt&quot;, &quot;n&quot;], optional): _description_. Defaults to "c".
            adf_autolag (Literal[&quot;AIC&quot;, &quot;BIC&quot;, &quot;t, optional): _description_. Defaults to "AIC".
            max_d (int, optional): Max order of differencing. Defaults to 5.
            p_max (int, optional): Max order for AR(p). Defaults to 24.
            ks_method (Literal[&quot;auto&quot;, &quot;exact&quot;, &quot;asymp&quot;], optional): Defaults to "exact".
            ks_alternative (str, optional): Defaults to "two-sided".
            acf_k (int, optional): Number of lags used for Autocorrelation Diagnostics and Ljung-Box Tets. Defaults to 24.
            plot (bool, optional):Defaults to False.
            plot_save_directory (Path, optional): Defaults to Path(ic.PATH_PRINTS).
            dpi (int, optional): Defaults to 300.

        Returns:
            Dict[str, Any]: _description_
        """
        wt_id = df[ic.WT_ID].iloc[0]
        pre_raw, post_raw = cls.split_pre_post(df, col=col, ts_det=ts_det, offset=offset, ts_col=ts_col)
        
        pre_raw_arr = np.asarray(pre_raw, dtype=float)
        post_raw_arr = np.asarray(post_raw, dtype=float)
        
        delta_median_raw = float(np.median(post_raw_arr) - np.median(pre_raw_arr))
        delta_cliffs_raw = float(cls.cliffs_delta(pre_raw_arr, post_raw_arr))
        
        eps = 1e-10
        delta_ln_sigma_raw = float( np.log( np.std(post_raw_arr, ddof=1) / (np.std(pre_raw_arr, ddof=1) + eps) ))
         
        # if pre_raw.size < 20 or post_raw.size < 20:
        #     raise ValueError(f"not enough samples")

        # diff_info = cls.difference_until_stationary(
        #     pre_raw,
        #     alpha=adf_alpha,
        #     regression=adf_regression,
        #     autolag=adf_autolag,
        #     max_d=max_d,
        # )
        # pre_diff = np.asarray(diff_info["x"], dtype=float)
        # d = int(diff_info["d"])
        
        # post_diff = np.diff(post_raw, n=d) if d > 0 else post_raw
        # post_diff = np.asarray(post_diff, dtype=float)
        # post_diff = post_diff[np.isfinite(post_diff)]
        
        # if pre_diff.size < 20 or post_diff.size < 20:
        #     raise ValueError(f"Too few samples after differencing")
        
        # order_table = cls.select_ar_order_aic(
        #     pre_diff,
        #     p_max=p_max,
        #     include_const=include_const,
        # )
        
        # p = cls._pick_best_p(order_table)
        
        # pw = cls.prewhitening_residuals(
        #     pre=pre_diff,
        #     post=post_diff,
        #     p=p,
        #     include_const=include_const,
        # )

        # resid_pre = np.asarray(pw.resid_pre, dtype=float)
        # resid_post = np.asarray(pw.resid_post, dtype=float)
        # resid_pre = resid_pre[np.isfinite(resid_pre)]
        # resid_post = resid_post[np.isfinite(resid_post)]
        
        # if resid_pre.size < 40 or resid_post.size < 40: # needed to approx. D_crit (Conover,1999)
        #     raise ValueError(f"Not enough samples. n_pre={resid_pre.size}, n_post={resid_post.size}")
        
        # resid_pre, resid_post, sigma_pre = cls.standardize_by_pre_sigma(resid_pre, resid_post,ddof=1)
            
        # wn_res_pre = cls.check_white_noise(resid_pre, lags=acf_k)
        # wn_res_post = cls.check_white_noise(resid_post, lags=acf_k)

        # ks_res = cls.ks_on_residuals(
        #     resid_pre,
        #     resid_post,
        #     method=ks_method,
        #     alternative=ks_alternative
        # )

        return {
            "signal": col,
            "ts_det": pd.to_datetime(ts_det),
            "offset": offset,
            # "n_pre_raw": pre_raw.size,
            # "n_post_raw": post_raw.size,
            # "d": d,
            # "adf": diff_info,
            # "order_table": order_table,
            # "p": p,
            # "include_const": include_const,
            # "sigma_pre": sigma_pre,
            # "ks": ks_res,
            # "pre_diag": wn_res_pre,
            # "post_diag": wn_res_post,
            #"pw_results": pw,
            # "adf_regression": adf_regression,
            ic.WT_ID: wt_id,
            "delta_median": delta_median_raw,
            "delta_ln_sigma": delta_ln_sigma_raw,
            "cliffs_delta": delta_cliffs_raw,
            # "plot_inputs": {
            #     "acf_k": int(acf_k),
            #     "pre_diff": pre_diff,
            #     "post_diff": post_diff,
            #     "resid_pre": resid_pre,
            #     "resid_post": resid_post,
            # }
        }
    
    @staticmethod
    def report_total(pipeline_results: Dict[str,Any], alpha: float= 0.05) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: "method","alternative","D","D_crit","p_value","alpha","n_pre","n_post":
        """
        # def _approx_d_crit(n1, n2, alpha):
        #     if alpha != 0.05:
        #         raise ValueError(f"approx. only for alpha=0.05")
        #     # for alpha=0.05 and n>40 or n1, n2 very large: source table of conover 1980
        #     # for this setting the approximation is the same (n1!=n2 or n1=n2)
        #     d_crit = 1.36 * np.sqrt((n1+n2)/(n1*n2))
        #     return d_crit
        
        #signal = pipeline_results["signal"]
        #D = pipeline_results["ks"]["ks_D"]
        #p = pipeline_results["ks"]["ks_p"]
        #alpha = alpha
        #n_pre = pipeline_results["ks"]["n_pre"]
        #n_post = pipeline_results["ks"]["n_post"]
        #D_crit = _approx_d_crit(n_pre, n_post, alpha)
        #alt = pipeline_results["ks"]["alternative"]
        #method = pipeline_results["ks"]["method"]
        delta = pipeline_results["cliffs_delta"]
        delta_median = pipeline_results["delta_median"]
        delta_ln_sigma = pipeline_results["delta_ln_sigma"]
        #diff_order_d = pipeline_results["d"]
        #reject_h0  = D > D_crit
        return pd.DataFrame([{
            "wt id": pipeline_results[ic.WT_ID],
            "ts center": pipeline_results["ts_det"],
            "signal": pipeline_results["signal"],
            #"ks test statistic": D,
            #"applied diff. order d": diff_order_d,
            "∆ median (raw)": delta_median,
            "∆ ln σ  (raw)": delta_ln_sigma,
            "cliff's delta (raw)": delta,
            #"ks pre resid. samples": n_pre,
            #"ks post resid. samples": n_post,
        }]).round(3)
    
    @staticmethod
    def report_stationarity(pipeline_results: Dict[str, Any], alpha:float=0.05) -> pd.DataFrame:
        pre_teststatistic = pipeline_results["adf"]["adf_pre"].test_stat
        pre_pvalue = pipeline_results["adf"]["adf_pre"].p_value
        pre_lag = pipeline_results["adf"]["adf_pre"].lag
        pre_nobs = pipeline_results["adf"]["adf_pre"].n_obs
        pre_is_stationary = pipeline_results["adf"]["stationary_pre"]
        
        post_teststatistic = pipeline_results["adf"]["adf_post"].test_stat
        post_pvalue = pipeline_results["adf"]["adf_post"].p_value
        post_lag = pipeline_results["adf"]["adf_post"].lag
        post_nobs = pipeline_results["adf"]["adf_post"].n_obs
        post_is_stationary = pipeline_results["adf"]["stationary_post"]
        d = pipeline_results["adf"]["d"]
        regression_model  = pipeline_results["adf_regression"]
        test_name="ADF test"
        
        
        
        return pd.DataFrame([{
            "wt id": pipeline_results[ic.WT_ID],
            "detection ts": pipeline_results["ts_det"],
            "signal": pipeline_results["signal"],
            
            "test method": test_name,
            "regression model": regression_model,
            "alpha": alpha,
            "pre test_stat": pre_teststatistic,
            "pre p value": pre_pvalue,
            "pre lag": pre_lag,
            "pre samples": pre_nobs,
            "reject h0 (before diff.)": pre_is_stationary,
            
            "post teststat": post_teststatistic,
            "post p value": post_pvalue,
            "post lag": post_lag,
            "post samples": post_nobs,
            "reject h0 (after diff.)": post_is_stationary,
            
            "d applied to pre and post": d
        }]).round(3)
    
    @staticmethod
    def report_AR_whitening(pipeline_results: Dict[str,Any]) -> pd.DataFrame:
        order_table = pipeline_results["order_table"]
        order = int(pipeline_results["p"])
        row = order_table[order_table["p"] == order]
        if row.empty:
            raise ValueError(f"Selected p={order} not found in order_table")
        row = row.iloc[0]
        
        method = str(row["prefer"]) # AIC or AICc
        aic_value = row[method]
        
        model = "AR(p)"
        phi = pipeline_results["pw_results"].phi # model params
        intercept = pipeline_results["pw_results"].intercept # const

        return pd.DataFrame([{
            "wt id": pipeline_results[ic.WT_ID],
            "detection ts": pipeline_results["ts_det"],
            "signal": pipeline_results["signal"],
            "order selection": method,
            "value": aic_value,
            "order p": order,
            "model": model,
            "model params.": phi,
            "intercept": intercept,
            
        }]).round(3)
        
    @staticmethod
    def report_standardization(pipeline_results:Dict[str, Any]) -> pd.DataFrame:
        sig_pre =  pipeline_results["sigma_pre"]
        
        return pd.DataFrame([{
            "wt id": pipeline_results[ic.WT_ID],
            "detection ts": pipeline_results["ts_det"],
            "signal": pipeline_results["signal"],
            "sigma_pre (for pre and post)": sig_pre,
        }]).round(3)
    
    @staticmethod
    def report_white_noise_before_ks_test(pipeline_results:Dict[str,Any]) -> pd.DataFrame:
        
        pre_is_white = pipeline_results["pre_diag"][0]
        pre_pvalue = pipeline_results["pre_diag"][1]
        
        post_is_white = pipeline_results["post_diag"][0]
        post_pvalue = pipeline_results["post_diag"][1]
        pre_post_alpha = pipeline_results["pre_diag"][2]
        pre_post_lags = pipeline_results["pre_diag"][3]
        pre_test_stat = pipeline_results["pre_diag"][4]
        post_test_stat = pipeline_results["post_diag"][4]
        test_name = "Ljung-Box-Test"
        return pd.DataFrame([{
            "wt id": pipeline_results[ic.WT_ID],
            "detection ts": pipeline_results["ts_det"],
            "signal": pipeline_results["signal"],
            "test method": test_name,
            "pre reject h0": pre_is_white,
            "pre test stat.": pre_test_stat,
            "pre p value": pre_pvalue,
            
            "post reject h0": post_is_white,
            "post test stat.": post_test_stat,
            "post p value": post_pvalue,
            
            "alpha": pre_post_alpha,
            "lags": pre_post_lags,
        }]).round(3)
    
    @staticmethod
    def report_windows(pipeline_results: Dict[str, Any]) -> pd.DataFrame:
        ts_det = pd.to_datetime(pipeline_results["ts_det"])
        offset = pd.to_timedelta(pipeline_results["offset"])
        dt = pd.Timedelta(minutes=10)
        pre_start = ts_det - offset
        pre_end = ts_det
        pre_n_samples = pipeline_results["n_pre_raw"]
        
        post_start = ts_det + dt
        post_end = (ts_det + dt) + offset
        post_n_samples = pipeline_results["n_post_raw"]
        signal = pipeline_results["signal"]
        wt_id = pipeline_results[ic.WT_ID]
        
        return pd.DataFrame([{
            "wt_id": wt_id,
            "signal": signal,
            "detection ts": pipeline_results["ts_det"],
            
            "pre start": pre_start,
            "pre end": pre_end,
            "pre samples": pre_n_samples,
            "post start": post_start,
            "post end": post_end,
            "post samples": post_n_samples 
        }]).round(3)
    
    #helpers
    def report_total_many(results: list[dict], alpha: float= 0.05) -> pd.DataFrame:
        return pd.concat([ks_test.report_total(r, alpha=alpha) for r in results], ignore_index=True)
    
    def report_stationarity_many(results: list[dict], alpha: float= 0.05) -> pd.DataFrame:
        return pd.concat([ks_test.report_stationarity(r, alpha=alpha) for r in results], ignore_index=True)
    
    def report_ar_many(results: list[dict]) -> pd.DataFrame:
        return pd.concat([ks_test.report_AR_whitening(r) for r in results], ignore_index=True)
    
    def report_std_many(results: list[dict]) -> pd.DataFrame:
        return pd.concat([ks_test.report_standardization(r) for r in results], ignore_index=True)
    
    def report_lb_many(results: list[dict]) -> pd.DataFrame:
        return pd.concat([ks_test.report_white_noise_before_ks_test(r) for r in results], ignore_index=True)
    
    def report_windows_many(results: list[dict]) -> pd.DataFrame:
        return pd.concat([ks_test.report_windows(r) for r in results], ignore_index=True)
    