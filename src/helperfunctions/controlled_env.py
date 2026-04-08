import pandas as pd
from enum import Enum
from helperfunctions import intern_constants as ic
from typing import Tuple, Optional, List, Dict, TypedDict, NotRequired, Any, Union
from helperfunctions.helper import load_feature_order
import numpy as np
from dataclasses import dataclass
from enum import StrEnum, auto
import os
from pathlib import Path

class Anom_Type(Enum):
    CONST_OFFSET = 1
    MULT_DRIFT = 2
    
class AnomCategory(Enum):
    ADD = auto()
    POINT = auto()
    MULT = auto()
    CORR = auto()

def _cat_enum(cat) -> AnomCategory:
    return cat if isinstance(cat, AnomCategory) else AnomCategory[str(cat)]

@dataclass
class AnomalySpec:
    """Available Attributes: \n
        - wt_id \n
        - category \n
        - primary_signal \n
        - secondary_signal \n
        - intensities \n
        - k_anoms 
    """
    wt_id: int
    category: AnomCategory
    primary_signal: str # The signal to manipulate
    secondary_signal: Optional[str] = None # required for corr anomaly
    intensities: Optional[List[float]] = None # parameters for anomaly injection
    k_anoms: int = 1 # Mult = 1, others = 3
    
@dataclass
class AnomalyPlan:
    """Global time window and synchronized event windows per anomaly category.\n
        Available Attributes:
        - ts0 \n
        - ts1 \n
        - gap \n
        - min_len \n
        - windows_by_cat \n
    """
    ts0: pd.Timestamp
    ts1: pd.Timestamp
    gap: pd.Timedelta
    min_len: pd.Timedelta
    windows_by_cat: Dict[AnomCategory, List[Tuple[pd.Timestamp, pd.Timestamp]]]

class AnomOverviewKeys(StrEnum):
    GT_ID = "gt_id"
    WT_ID = "wt_id"
    ANOM_ID = "anom_id"
    CATEGORY = "category"
    SIGNALS = "signals"
    ANOM_TYPE = "anom_type"
    WINDOW_IDX = "window_idx"
    TS_START = "ts_start"
    TS_END = "ts_end"
    INTENSITY = "intensity"
    IOU = "iou"
    INTERSECTION_LEN = "intersection_len"
    GT_LEN = "gt_len"
    COVERAGE = "coverage"
    LATENCY = "latency"
    
class AnomOverviewDict(TypedDict):
    gt_id : NotRequired[int]
    wt_id : NotRequired[int]
    anom_id: NotRequired[int]
    category : NotRequired[AnomCategory]
    signals : NotRequired[str]
    anom_type : NotRequired[str]
    window_idx : NotRequired[int]
    ts_start : NotRequired[pd.Timestamp]
    ts_end : NotRequired[pd.Timestamp]
    intensity : NotRequired[float]
    iou: NotRequired[float]
    intersection_len: NotRequired[int]
    gt_len: NotRequired[int]
    coverage: NotRequired[float]
    latency: NotRequired[int]

class Part1:
    STEP = pd.Timedelta(minutes=10)
    
    @classmethod
    def _create_event_windows(cls,
                           ts0: pd.Timestamp,
                           ts1: pd.Timestamp,
                           k: int,
                           min_len: pd.Timedelta,
                           gap: pd.Timedelta
                            ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """ Creates k windows [s_i , e_i] inside the span of [ts0, ts1].
            Each at least min_len long and separated by a gap.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: _description_
        """
        total = (ts1 - ts0)
        usable = total - (k-1) * gap
        if usable < k * min_len:
            raise ValueError(f"Span too short for k windows. usable:{usable} <= {k*min_len}")
        
        # state usable > k*min_len
        base = usable / k
        
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = ts0
        
        for _ in range(k):
            start = cur
            end = min(start + max(min_len, base), ts1) # capped to ts1
            windows.append((start, end))
            cur = end + gap
        
        # make sure the last window does not exceed ts1
        windows[-1] = (windows[-1][0], min(windows[-1][1], ts1))
        
        return windows

    @classmethod
    def _check_gaps(cls,
                    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
                    gap: pd.Timedelta
                    ) -> None:
        """Checks that consecutive windows [s1, e1], [s2, e2] fulfill s2 - e1 >= gap
        """
        windows = sorted(windows, key=lambda w: w[0])
        for (_,e1), (s2, _) in zip(windows, windows[1:]):
            assert (s2 - e1) >= gap, f"gap error: {(s2 - e1)} < {gap}"

    @classmethod
    def _assert_no_overlap_across_categories(cls,
                                             windows_by_cat: Dict[AnomCategory, 
                                                                  List[Tuple[pd.Timestamp, 
                                                                             pd.Timestamp]]]
                                             ) -> None:
        items = []
        for cat, spans in windows_by_cat.items():
            for start, end in spans:
                items.append((cat, pd.to_datetime(start), pd.to_datetime(end)))
        
        items.sort(key=lambda x: x[1])
        
        for i in range(len(items)):
            cat_i, start_i, end_i = items[i]
            for j in range(i+1, len(items)):
                cat_j, start_j , end_j = items[j]
                if cat_i == cat_j:
                    continue
                if max(start_i, start_j) <= min(end_i, end_j):
                    raise AssertionError(f"Overlapping anomalies detected. Between {cat_i.name} and {cat_j.name}:"
                                         f"[{start_i}..{end_i}] vs [{start_j}..{end_j}]")
    
    @classmethod
    def infer_windows_by_category_from_specs(
        cls,
        specs: List[AnomalySpec],
        ) -> Dict[AnomCategory, int]:
        if not specs:
            raise ValueError("specs is empty.")
        
        n_windows_by_cat : Dict[AnomCategory, int] = {}
        
        for spec in specs:
            cat = spec.category
            if spec.intensities is None:
                raise ValueError(f"Spec for wt_id={spec.wt_id}, category={cat.name} has no intensities.")
            if cat is AnomCategory.MULT:
                n_spec= 1
            else:
                n_spec = len(spec.intensities)
            
            if cat in n_windows_by_cat and n_windows_by_cat[cat] != n_spec:
                raise ValueError(f"inconsistent number of intensities for category {cat.name}")
            
            n_windows_by_cat[cat] = n_spec
        
        for cat in AnomCategory:
            n_windows_by_cat.setdefault(cat, 0)
        
        return n_windows_by_cat
    
    @classmethod
    def build_plan(cls,
                   start: str,
                   end: str,
                   gap_in_hours: int,
                   min_len_hours: int,
                   k_add: int = 3,
                   k_point: int = 3,
                   k_mult: int = 1,
                   k_corr: int = 3,
                   return_cat_windows: bool = False,
                ) -> Union[AnomalyPlan, Tuple[AnomalyPlan, pd.DataFrame]]:
        
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        gap = pd.to_timedelta(gap_in_hours, "h")
        min_len = pd.to_timedelta(min_len_hours, "h")
        
        k_by_category: Dict[AnomCategory, int] = {
            AnomCategory.ADD: k_add,
            AnomCategory.POINT: k_point,
            AnomCategory.MULT: k_mult,
            AnomCategory.CORR: k_corr,
        }
        active = [(cat,k) for cat, k in k_by_category.items() if k > 0]
        
        def _req_len(k:int) -> pd.to_timedelta:
            return k * min_len + max(0,k-1)*gap
        
        total_required = sum((_req_len(k) for _, k in active), start=pd.Timedelta(0))
        total_gaps = max(0, len(active)-1)*gap
        total_available = (end - start)
        
        if total_required + total_gaps > total_available:
            raise ValueError("time period (start,end) is to small for all anomaly types")

        order = [AnomCategory.ADD, AnomCategory.POINT, AnomCategory.MULT, AnomCategory.CORR]
        windows_by_category: Dict[AnomCategory, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        curr = start
        
        for category in order:
            k = k_by_category.get(category,0)
            if k <=0:
                continue
            
            req_total_time = _req_len(k)
            sub_start = curr
            sub_end = min(sub_start + req_total_time, end)
            
            raw_windows = cls._create_event_windows(ts0=sub_start,
                                                    ts1=sub_end,
                                                    k=k,
                                                    min_len=min_len,
                                                    gap=gap)
            
            windows_by_category[category] = ([(e,e) for (_,e) in raw_windows] if category == AnomCategory.POINT else raw_windows)
            
            curr = sub_end + gap
        
        cls._assert_no_overlap_across_categories(windows_by_category)
        
        plan = AnomalyPlan(ts0=start, ts1=end, gap=gap, min_len=min_len, windows_by_cat=windows_by_category)
        
        if not return_cat_windows:
            return plan
        
        rows = []
        for cat, spans in windows_by_category.items():
            for idx, (s,e) in enumerate(spans):
                rows.append({
                    AnomOverviewKeys.CATEGORY: cat.name,
                    AnomOverviewKeys.WINDOW_IDX: int(idx),
                    AnomOverviewKeys.TS_START: pd.to_datetime(s),
                    AnomOverviewKeys.TS_END: pd.to_datetime(e)
                })
        win_df = (pd.DataFrame(rows)
                  .sort_values([AnomOverviewKeys.CATEGORY, AnomOverviewKeys.WINDOW_IDX, AnomOverviewKeys.TS_START])
                  .reset_index(drop=True))
        
        return plan, win_df
    
    
    @classmethod
    def _gt_union_per_wt(cls, df_gt: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for wt, group in df_gt.groupby(AnomOverviewKeys.WT_ID, sort=False):
            spans = sorted(zip(pd.to_datetime(group[AnomOverviewKeys.TS_START]),
                               pd.to_datetime(group[AnomOverviewKeys.TS_END])))
            merged = []
            for start, end in spans:
                if not merged or start > merged[-1][1]:
                    merged.append([start,end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            
            for j, (start,end) in enumerate(merged):
                rows.append({
                    AnomOverviewKeys.GT_ID: j,
                    AnomOverviewKeys.WT_ID: int(wt),
                    AnomOverviewKeys.CATEGORY: "WT_MEAN",
                    AnomOverviewKeys.SIGNALS: "Mean",
                    AnomOverviewKeys.WINDOW_IDX: j,
                    AnomOverviewKeys.TS_START: start,
                    AnomOverviewKeys.TS_END: end,
                    AnomOverviewKeys.ANOM_TYPE: "UNION",
                    AnomOverviewKeys.INTENSITY: float('nan')
                })
        return pd.DataFrame(rows)
    
    @classmethod
    def _last_sample_in_window(cls,
                               df: pd.DataFrame,
                               start: pd.Timestamp,
                               end: pd.Timestamp,
                               ) -> pd.Timestamp:
        """ Returns the last timestamp in window [start, end] found in df[ic.TS_COL].
            This ts will be used to inject a point anomaly
        """
        tss = df[(df[ic.TS_COL] >= start) & (df[ic.TS_COL] <= end)][ic.TS_COL].sort_values()
        if tss.empty:
            raise ValueError("The given window has no samples")
        
        return tss.iloc[-1]
    
    @classmethod
    def apply_plan_and_build(cls,
                             df_raw_sigs: pd.DataFrame,
                             specs: List[AnomalySpec],
                             plan: AnomalyPlan,
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Applies the anomaly plan to a copy of df_raw_sigs.
        Returns:
            (modified_df, ground_truth_table) with columns:
            [gt_id, wt_id, category, signals, anom_type, window_idx, ts_start, ts_end, intensity]
        """
        df = df_raw_sigs.copy(deep=True)
        rows: List[AnomOverviewDict] = []
        gt_id = 0
        
        for spec in specs:
            windows = plan.windows_by_cat[spec.category]
            
            if spec.category in (AnomCategory.ADD, AnomCategory.POINT, AnomCategory.CORR):
                assert spec.intensities and len(spec.intensities) == len(windows), (f"Missing intensities for ADD/POINT/CORR.\n"
                f"intensities:{spec.intensities} != len(windows):{len(windows)}")

            if spec.category == AnomCategory.MULT: #MULT has only one Window
                assert spec.intensities and len(spec.intensities) == 1, (f"MULT needs exactly one intensity."
                f"Given intensities: {spec.intensities}")
            
            if spec.category == AnomCategory.CORR:
                assert spec.secondary_signal, "CORR requires a secondary_signal."
                
            for idx, (start, end) in enumerate(windows):
                match spec.category:
                    case AnomCategory.ADD:
                        df = Inject_Anomalies.inject_anom_type(
                            df=df,
                            start_ts=start,
                            end_ts=end,
                            slope=spec.intensities[idx],
                            target_col= spec.primary_signal,
                            operator=Anom_Type.CONST_OFFSET,
                            ts_col=ic.TS_COL,
                            wt_id=spec.wt_id,
                        )
                        rows.append({
                            AnomOverviewKeys.GT_ID: gt_id,
                            AnomOverviewKeys.WT_ID: spec.wt_id,
                            AnomOverviewKeys.CATEGORY: spec.category.name,
                            AnomOverviewKeys.SIGNALS: spec.primary_signal,
                            AnomOverviewKeys.ANOM_TYPE: "ADD",
                            AnomOverviewKeys.WINDOW_IDX: idx,
                            AnomOverviewKeys.TS_START:start,
                            AnomOverviewKeys.TS_END: end,
                            AnomOverviewKeys.INTENSITY: spec.intensities[idx]
                        })
                    case AnomCategory.POINT:
                        ts_last = cls._last_sample_in_window(df, start, end)
                        df = Inject_Anomalies.inject_anom_type(
                            df=df,
                            start_ts=ts_last,
                            end_ts=ts_last,
                            slope=spec.intensities[idx],
                            target_col=spec.primary_signal,
                            operator=Anom_Type.CONST_OFFSET,
                            ts_col=ic.TS_COL,
                            wt_id=spec.wt_id,
                        )
                        rows.append({
                            AnomOverviewKeys.GT_ID: gt_id,
                            AnomOverviewKeys.WT_ID: spec.wt_id,
                            AnomOverviewKeys.CATEGORY: spec.category.name,
                            AnomOverviewKeys.SIGNALS: spec.primary_signal,
                            AnomOverviewKeys.ANOM_TYPE: "POINT",
                            AnomOverviewKeys.WINDOW_IDX: idx,
                            AnomOverviewKeys.TS_START:ts_last,
                            AnomOverviewKeys.TS_END: ts_last,
                            AnomOverviewKeys.INTENSITY: spec.intensities[idx]
                        })
                    case AnomCategory.MULT:
                        df = Inject_Anomalies.inject_anom_type(
                            df=df,
                            start_ts=start,
                            end_ts=end,
                            slope=spec.intensities[0],
                            target_col= spec.primary_signal,
                            operator=Anom_Type.MULT_DRIFT,
                            ts_col=ic.TS_COL,
                            wt_id=spec.wt_id,
                        )
                        rows.append({
                            AnomOverviewKeys.GT_ID: gt_id,
                            AnomOverviewKeys.WT_ID: spec.wt_id,
                            AnomOverviewKeys.CATEGORY: spec.category.name,
                            AnomOverviewKeys.SIGNALS: spec.primary_signal,
                            AnomOverviewKeys.ANOM_TYPE: "MULT",
                            AnomOverviewKeys.WINDOW_IDX: idx,
                            AnomOverviewKeys.TS_START:start,
                            AnomOverviewKeys.TS_END: end,
                            AnomOverviewKeys.INTENSITY: spec.intensities[0]
                        })
                    case AnomCategory.CORR:
                        rho_target = spec.intensities[idx]
                        df = Inject_Anomalies.inject_corr_anomaly(
                            df=df,
                            signal1=spec.primary_signal,
                            signal2=spec.secondary_signal,
                            window=(start, end),
                            corr_target=rho_target,
                            seed=32,
                            ts_col=ic.TS_COL,
                            wt_id=spec.wt_id,
                            edge_taper_frac=0.0,
                        )
                        rows.append({
                            AnomOverviewKeys.GT_ID: gt_id,
                            AnomOverviewKeys.WT_ID: spec.wt_id,
                            AnomOverviewKeys.CATEGORY: spec.category.name,
                            AnomOverviewKeys.SIGNALS: f"{spec.primary_signal}|{spec.secondary_signal}",
                            AnomOverviewKeys.ANOM_TYPE: "CORR",
                            AnomOverviewKeys.WINDOW_IDX: idx,
                            AnomOverviewKeys.TS_START:start,
                            AnomOverviewKeys.TS_END: end,
                            AnomOverviewKeys.INTENSITY: rho_target
                        })
                    case _:
                        raise ValueError(f"Non known AnomCategory.")
                gt_id +=1
        gt = pd.DataFrame(rows, columns=[AnomOverviewKeys.GT_ID, AnomOverviewKeys.WT_ID, AnomOverviewKeys.CATEGORY, AnomOverviewKeys.SIGNALS, AnomOverviewKeys.ANOM_TYPE, AnomOverviewKeys.WINDOW_IDX, AnomOverviewKeys.TS_START, AnomOverviewKeys.TS_END, AnomOverviewKeys.INTENSITY])
        
        return df, gt
    
    @classmethod
    def prepare_df_eval_threshold_computing(cls,
                                            df_eval: pd.DataFrame,
                                            include_mean: bool = True,
                                            include_signal_wise_re: bool= False,
                                            pick_signals_re: Optional[List[str]] = None,
                                            ) -> pd.DataFrame:
        """ df_eval is the output of training_lib.eval_model.
            This Function converts df_eval into a table useful to determine thresholds .
            
            Returns:
            DataFrame: ["Date and time", "WT_ID", "Signal", "Loss"].
            """
        meta_cols = [ic.TS_COL, ic.WT_ID]
        parts = []
        
        if include_mean:
            if ic.MEAN_LOSS_PER_SAMPLE not in df_eval.columns:
               raise KeyError(f"{ic.MEAN_LOSS_PER_SAMPLE} is not in df_eval")
           
            df_mean = (
               df_eval[meta_cols + [ic.MEAN_LOSS_PER_SAMPLE]].copy()
               .rename(columns={ic.MEAN_LOSS_PER_SAMPLE: ic.RE_COL})
            )

            df_mean[ic.SIGNAL_COL] = "Mean"
            parts.append(df_mean[meta_cols + [ic.SIGNAL_COL, ic.RE_COL]])

        
        if include_signal_wise_re:
            if pick_signals_re is not None:
                for sig in pick_signals_re:
                    if sig not in df_eval.columns:
                        raise ValueError(f"The signal:{sig} in pick_signals_re is not in df_eval")
            
            res_cols = list(pick_signals_re) if pick_signals_re is not None else [c for c in df_eval.columns if c.startswith(ic.RE_PREFIX)]
            if res_cols:
                new_df = df_eval[meta_cols + res_cols].melt(
                    id_vars=meta_cols,
                    var_name=ic.SIGNAL_COL,
                    value_name=ic.RE_COL
                )
                #remove PREFIX RE_
                new_df[ic.SIGNAL_COL] = new_df[ic.SIGNAL_COL].str.replace(f"^{ic.RE_PREFIX}", "", regex=True)
                parts.append(new_df[meta_cols + [ic.SIGNAL_COL, ic.RE_COL]])

        if not parts:
            raise ValueError(f"Nothing selected in parts")
        
        df = pd.concat(parts, ignore_index=True)
        df[ic.TS_COL] = pd.to_datetime(df[ic.TS_COL])
        df = df.sort_values([ic.WT_ID, ic.SIGNAL_COL, ic.TS_COL]).reset_index(drop=True)
        return df
    
    @classmethod
    def compute_val_sigma(cls,
                            re_val_df: pd.DataFrame
                            ) -> pd.DataFrame:
        """Computes per (wt_id, signal) std and mean on          
        reconstruction errors. \n
        re_val_df is the output of prepare_df_eval_threshold_computing().
        
        Returns:
        DataFrame: [wt_id, signal,mu, sigma]
        """
        grouped_df = (re_val_df
                        .groupby([ic.WT_ID, ic.SIGNAL_COL])[ic.RE_COL]
                        .agg(mu="mean",
                           sigma=lambda x: x.std(ddof=0)
                        )
                        .reset_index()
                      )
        
        eps = 1e-12
        grouped_df["sigma"] = grouped_df["sigma"].replace(0.0, eps).fillna(eps)
        
        return grouped_df[[ic.WT_ID, ic.SIGNAL_COL,"mu", "sigma"]]

    # new implementation: ###################
    @classmethod
    def build_threshold_grid(cls,
                             sigma_df: pd.DataFrame,
                             k_values: List[float],
                             ) -> pd.DataFrame:
        """

        Returns:
            pd.DataFrame: [ic.WT_ID, ic.SIGNAL_COL,"mu", "sigma", "k", "threshold"]
        """
        if sigma_df is None or sigma_df.empty:
            raise ValueError("sigma_df is empty")
        if "sigma" not in sigma_df.columns:
            raise KeyError("The column sigma is not in sigma_df")
        if "mu" not in sigma_df.columns:
            raise KeyError("The column mu is not in sigma_df")
        if ic.WT_ID not in sigma_df.columns:
            raise KeyError(f"The column {ic.WT_ID} is not in sigma_df ")
        if ic.SIGNAL_COL not in sigma_df.columns:
            raise KeyError(f"The columns {ic.SIGNAL_COL} is not in sigma_df")
        if not k_values:
            raise ValueError("k_values is empty")
        
        k_unique = sorted({float(k) for k in k_values})
        k_df = pd.DataFrame({"k": k_unique})
        
        sigma_df = sigma_df[[ic.WT_ID, ic.SIGNAL_COL,"mu", "sigma"]].copy()
         
        sigma_df["_tmp_key"] = 1
        k_df["_tmp_key"] = 1
        
        grid = (
            sigma_df.merge(k_df, on="_tmp_key", how="outer")
            .drop(columns="_tmp_key")
        )
        
        grid["threshold"] =grid["mu"]+  grid["k"] * grid["sigma"]
        
        grid = (
            grid[[ic.WT_ID, ic.SIGNAL_COL,"mu", "sigma", "k", "threshold"]]
            .sort_values([ic.WT_ID, ic.SIGNAL_COL, "k"])
            .reset_index(drop=True)
        )
        return grid
    
    @classmethod
    def _count_fp_for_wt(
        cls,
        ts_index: pd.DatetimeIndex,
        is_pos: np.ndarray,
        neg_spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
        merge_gap_steps: int = 6
    ) -> int:
        """Counts False Alarm Episodes (FP)

        Args:
            ts_index (pd.DatetimeIndex): sorted ts-index of eval. samples (only mean reconstruction error)
            
            is_pos (np.ndarray): boolean array, same length as ts_index. True = Hit (RE >= threshold)
            
            neg_spans (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of negative windows (complement of gt-union as [(start, end),...])
            
            merge_gap_steps (int): max. length of a gap. Fps within the gap will be merged to a FP episode.
            
        """
        if ts_index.empty or len(neg_spans) == 0:
            return 0

        n = len(ts_index)
        is_fp = np.zeros(n, dtype=bool)
        
        for (start, end) in neg_spans:
            left = ts_index.searchsorted(start, side="left")
            right = ts_index.searchsorted(end, side="right")
            
            if right <= left:
                continue
            
            is_fp[left:right] |= is_pos[left:right]
            
        hit_indices = np.flatnonzero(is_fp)
        
        if hit_indices.size == 0:
            return 0
        
        if merge_gap_steps < 0:
            raise ValueError("merge_gap_steps must be >= 0")
        
        fp_events = 0
        
        curr_start_idx = None
        curr_end_idx = None
        
        for idx in hit_indices:
            if curr_start_idx is None:
                curr_start_idx = idx
                curr_end_idx = idx
                continue
            
            gap = ts_index[idx] - ts_index[curr_end_idx]
            
            if gap < merge_gap_steps* cls.STEP:
                curr_end_idx = idx
            else:
                fp_events += 1
                curr_start_idx = idx
                curr_end_idx = idx
        
        if curr_start_idx is not None:
            fp_events += 1
        
            
        return fp_events
    
    @classmethod
    def eval_events_over_k(cls,
                           wt_sig_loss_df: pd.DataFrame,
                           gt_union_df: pd.DataFrame,
                           gt_events_df: pd.DataFrame,
                           thresholds_grid_df: pd.DataFrame,
                           step: pd.Timedelta = STEP,
                           mean_label: str = "Mean",
                           fp_merge_gap_steps: int = 6,
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Event-based evaluation for all wts and k-values.

        Args:
            wt_sig_loss_df (pd.DataFrame): From prepare_df_eval_threshold_computing()
            gt_union_df (pd.DataFrame): from _gt_union_per_wt
            gt_events_df (pd.DataFrame): from apply_plan_and_build()
            thresholds_grid_df (pd.DataFrame): required cols [ic.WT_ID, ic.SIGNAL_COL, k, sigma, threshold], SIGNAL_COL == mean_label
            step (pd.Timedelta, optional): Defaults to STEP.
            mean_label (str, optional): Defaults to "Mean".
            fp_merge_gap_steps (int): Neighbored FPs within the gap will be treated as one FP.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: per_wt_k_df: [ic.WT_ID, k , sigma, threshold, tp, fn, fp, precision, recall, f1, FAR_per_day, latency],\n per_event_df: [ic.WT_ID, k , sigma, threshold, anom_type, signals, ts_start, ts_end, latency, intensity]
        """
        required_loss_cols = {ic.TS_COL, ic.WT_ID, ic.SIGNAL_COL, ic.RE_COL}
        if not required_loss_cols.issubset(wt_sig_loss_df.columns):
            missing = required_loss_cols - set(wt_sig_loss_df.columns)
            raise KeyError(f"wt_sig_loss_df missing columns: {missing}")
        
        required_thresh_cols = {ic.WT_ID, ic.SIGNAL_COL, "k","mu", "sigma", "threshold"}
        if not required_thresh_cols.issubset(thresholds_grid_df.columns):
            missing = required_thresh_cols - set(thresholds_grid_df.columns)
            raise KeyError(f"thresholds_grid_df missing columns: {missing}")
        
        if AnomOverviewKeys.WT_ID not in gt_union_df.columns:
            raise KeyError(f"{AnomOverviewKeys.WT_ID} is missing in gt_union_df")
        
        if AnomOverviewKeys.WT_ID not in gt_events_df.columns:
            raise KeyError(f"{AnomOverviewKeys.WT_ID} is missing in gt_events_df")
        
        if not isinstance(step, pd.Timedelta):
            step = pd.to_timedelta(step)
        
        
        if thresholds_grid_df is None or thresholds_grid_df.empty:
            raise ValueError("thresholds_grid_df is empty")
        

        ## needed for False Alarm Rate per Year
        all_ts = pd.to_datetime(wt_sig_loss_df[ic.TS_COL], errors="raise")
        if all_ts.empty:
            raise RuntimeError("wt_sig_loss_df has no timestamps")
        
        start_global = all_ts.min().floor("D")
        end_global = all_ts.max().floor("D")
        inclusive_span =  (end_global - start_global) + pd.Timedelta(days=1)
        
        total_days = float(inclusive_span / pd.Timedelta(days=1))
        n_days = max(1.0, total_days)
        
        
        #Threshold map (WT_ID, k)
        thresh_mean = thresholds_grid_df[
            thresholds_grid_df[ic.SIGNAL_COL] == mean_label
        ].copy()
        
        if thresh_mean.empty:
            raise RuntimeError(
                f"thresholds_grid_df has no entries"
            )
        
        k_values = sorted(thresh_mean["k"].dropna().astype(float).unique().tolist())

        if len(k_values) == 0:
            raise RuntimeError("thresholds_grid_df has no entries")

        thresh_map: Dict[Tuple[int, float], Tuple[float, float,float]] = {}
        
        for _, row in thresh_mean.iterrows():
            wt = row[ic.WT_ID]
            k_val = row["k"]
            mu_val = row["mu"]
            sigma_val = row["sigma"]
            theta_val = row["threshold"]
            key = (wt, k_val)
            
            if key in thresh_map:
                raise RuntimeError(f"Duplicate threshold entry found for WT_ID={wt}, k={k_val} in thresholds_grid_df")
           
            thresh_map[key] = (mu_val, sigma_val, theta_val)
        
        loss_mean = wt_sig_loss_df[
            wt_sig_loss_df[ic.SIGNAL_COL] == mean_label
        ].copy()
        
        loss_mean[ic.TS_COL] = pd.to_datetime(loss_mean[ic.TS_COL], errors="raise")
        loss_mean[ic.WT_ID] = loss_mean[ic.WT_ID].astype(int)
        
        wt_ids_loss = set(loss_mean[ic.WT_ID].unique().tolist())
        wt_ids_union = set(gt_union_df[AnomOverviewKeys.WT_ID].astype(int).unique().tolist())
        wt_ids_events = set(gt_events_df[AnomOverviewKeys.WT_ID].astype(int).unique().tolist())
        
        if not wt_ids_union.issubset(wt_ids_loss):
            raise ValueError(f"wt_ids_union:{wt_ids_union} is not a subset of wt_ids_loss:{wt_ids_loss}")
        if not wt_ids_events.issubset(wt_ids_loss):
            raise ValueError(f"wt_ids_events:{wt_ids_events} is not a subset of wt_ids_loss:{wt_ids_loss}")
        if not wt_ids_union.issubset(wt_ids_events):
            raise ValueError(f"wt_ids_union:{wt_ids_union} is not a subset of wt_ids_events")
        
        all_wt_ids = sorted(wt_ids_events)
        
        
        gt_union_df = gt_union_df.copy()
        gt_union_df[AnomOverviewKeys.WT_ID] = gt_union_df[AnomOverviewKeys.WT_ID].astype(int)
        gt_union_df[AnomOverviewKeys.TS_START] = pd.to_datetime(gt_union_df[AnomOverviewKeys.TS_START], errors="raise")
        gt_union_df[AnomOverviewKeys.TS_END]= pd.to_datetime(gt_union_df[AnomOverviewKeys.TS_END], errors="raise")
        
        pos_spans_per_wt: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        
        for wt, group in gt_union_df.groupby(AnomOverviewKeys.WT_ID, sort=False):
            wt = int(wt)
            spans = list(
                zip(
                    group[AnomOverviewKeys.TS_START].tolist(),
                    group[AnomOverviewKeys.TS_END].tolist()
                )
            )
            pos_spans_per_wt[wt] = spans
        
        gt_events_df = gt_events_df.copy()
        gt_events_df[AnomOverviewKeys.WT_ID] = gt_events_df[AnomOverviewKeys.WT_ID].astype(int)
        gt_events_df[AnomOverviewKeys.TS_START] = pd.to_datetime(
            gt_events_df[AnomOverviewKeys.TS_START], errors="raise"
        )
        gt_events_df[AnomOverviewKeys.TS_END] = pd.to_datetime(
            gt_events_df[AnomOverviewKeys.TS_END], errors="raise"
        )
        
        events_per_wt: Dict[int, pd.DataFrame] = {}
        for wt, group in gt_events_df.groupby(AnomOverviewKeys.WT_ID, sort=False):
            events_per_wt[int(wt)]= group.sort_values(AnomOverviewKeys.TS_START).reset_index(drop=True)
        
        per_wt_k_rows: List[Dict[str, Any]] = []
        per_event_rows: List[Dict[str, Any]] = []
        
        for wt in all_wt_ids:
            wt = int(wt)
            
            loss_wt = loss_mean[loss_mean[ic.WT_ID] == wt].copy()
            loss_wt = loss_wt.sort_values(ic.TS_COL)
            
            ts_index = pd.DatetimeIndex(loss_wt[ic.TS_COL])
            re_vals = loss_wt[ic.RE_COL].astype(float).to_numpy(copy=False)
            
            if ts_index.empty:
                raise RuntimeError(f"No evaluation data found for wt_id={wt}.")
            
            span_start = ts_index.min()
            span_end = ts_index.max()
            
            pos_spans = pos_spans_per_wt.get(wt, [])
            neg_spans = cls._complement_intervalls(pos_spans, span_start, span_end)
            
            events_wt = events_per_wt.get(wt, gt_events_df.head(0))

            for k in k_values:
                key = (wt, k)
                if key not in thresh_map:
                    raise RuntimeError(f"Missing threshold for wt={wt}, k={k}"
                                       "thresholds_grid_df mus contain one row per (WT, 'Mean', k)")
            
                mu_val, sigma_val, theta_val = thresh_map[key]
                is_pos = (re_vals >= theta_val)
                
                fp = cls._count_fp_for_wt(ts_index, is_pos, neg_spans, fp_merge_gap_steps)
                
                tp = 0
                fn = 0
                latencies_for_mean: List[float] = []
                
                if not events_wt.empty:
                    for _, evt in events_wt.iterrows():
                        evt_start = evt[AnomOverviewKeys.TS_START]
                        evt_end =evt[AnomOverviewKeys.TS_END]
                        
                        left = ts_index.searchsorted(evt_start, side="left")
                        right = ts_index.searchsorted(evt_end, side="right")
                        
                        left= max(left, 0)
                        right = min(right, len(ts_index))
                        
                        latency_val: float = np.nan
                        hit = False
                        
                        if right > left:
                            local_hits = np.flatnonzero(is_pos[left:right])
                            if local_hits.size > 0:
                                idx_first = left + local_hits[0]
                                hit_ts = ts_index[idx_first]
                                
                                delta = hit_ts - evt_start
                                steps = max(delta / step, pd.Timedelta(0) / step)
                                latency_val = float(steps)
                                hit = True
                                
                        if hit:
                            tp +=1
                            latencies_for_mean.append(latency_val)
                        else:
                            fn += 1
                        
                        per_event_rows.append(
                            {
                                ic.WT_ID: wt,
                                "k": k,
                                "mu": mu_val,
                                "sigma": sigma_val,
                                "threshold": theta_val,
                                AnomOverviewKeys.ANOM_TYPE: evt.get(AnomOverviewKeys.ANOM_TYPE, None),
                                AnomOverviewKeys.SIGNALS: evt.get(AnomOverviewKeys.SIGNALS, None),
                                AnomOverviewKeys.TS_START: evt_start,
                                AnomOverviewKeys.TS_END: evt_end,
                                AnomOverviewKeys.LATENCY: latency_val,
                                AnomOverviewKeys.INTENSITY: evt.get(AnomOverviewKeys.INTENSITY, np.nan),
                            }
                        )
                
                
                precision = tp / float(tp + fp) if tp+fp > 0 else 0.0
                recall = tp / float(tp+fn) if tp + fn > 0 else 0.0
                f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                latency_mean = (float(np.mean(latencies_for_mean)) if len(latencies_for_mean) >0 else np.nan)
                
                far_per_day = fp / n_days
                
                per_wt_k_rows.append(
                    {
                        ic.WT_ID: wt,
                        "k": k,
                        "mu": mu_val,
                        "sigma":sigma_val,
                        "threshold": theta_val,
                        "tp": tp,
                        "fn": fn,
                        "fp": fp,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "FAR_per_day": far_per_day,
                        "latency_mean": latency_mean,
                    }
                )
        per_wt_k_df = (
            pd.DataFrame(per_wt_k_rows)
            .sort_values([ic.WT_ID, "k"])
            .reset_index(drop=True)
        )
        
        per_event_df = (
            pd.DataFrame(per_event_rows)
            .sort_values([ic.WT_ID, AnomOverviewKeys.TS_START, "k"])
            .reset_index(drop=True)
        )
        
        return per_wt_k_df, per_event_df
    
    
    @classmethod
    def select_k_per_wt(
        cls,
        per_wt_k_df: pd.DataFrame,
        save_dir: Optional[Path] = None,
        csv_name: str = "select_k_per_wt.csv"
        ) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: [ic.WT_ID,k,sigma,threshold,tp,fn,fp,precision,recall,f1, FAR_per_day,latency_mean],
        """
        if per_wt_k_df is None or per_wt_k_df.empty:
            raise ValueError("per_wt_k_df is empty.")
        
        df = per_wt_k_df.copy()
        
        df[ic.WT_ID] = pd.to_numeric(df[ic.WT_ID], errors="raise").astype(int)
        df["k"] = pd.to_numeric(df["k"], errors="raise")
        
        numeric_cols = [
            "mu",
            "sigma",
            "threshold",
            "tp",
            "fn",
            "fp",
            "precision",
            "recall",
            "f1",
            "FAR_per_day",
            "latency_mean",
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="raise")
            else:
                raise KeyError(f"col:{col} not in per_wt_k_df")
        
        best_rows: List[Dict[str,Any]] = []
        candidates_list: List[pd.DataFrame] = []
        
        for wt, group in df.groupby(ic.WT_ID, sort=True):
            wt = int(wt)
            group = group.copy()
            
            if "latency_mean" in group.columns:
                latency = group["latency_mean"]
                latency = latency.mask(~np.isfinite(latency), np.inf)
                group["latency_mean"] = latency
                
            group_sorted = group.sort_values(
                # ["f1","k", "latency_mean", "recall", "precision" ],
                ["f1", "k", "fp","latency_mean" ],
                ascending=[False ,True, True, True]
                
            ).reset_index(drop=True)
            
            group_sorted[ic.WT_ID] = wt
            
            candidates_list.append(group_sorted)
            best = group_sorted.iloc[0]
            
            best_rows.append(
                {
                    ic.WT_ID: wt,
                    "k": best["k"],
                    "mu": best["mu"],
                    "sigma": best["sigma"],
                    "threshold": best["threshold"],
                    "tp": best["tp"],
                    "fn": best["fn"],
                    "fp": best["fp"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                    "FAR_per_day": best["FAR_per_day"],
                    "latency_mean": best["latency_mean"],
                }
            )
        
        best_k_df = (
            pd.DataFrame(best_rows)
            .sort_values(ic.WT_ID)
            .reset_index(drop=True)
        )
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            candidates_df = pd.concat(candidates_list, ignore_index=True)
            candidates_path = Path(save_dir) / csv_name
            candidates_df.to_csv(candidates_path, index=False)
        
        return best_k_df
    
    
    
    @classmethod
    def build_target_table(
        cls,
        per_event_df: pd.DataFrame,
        best_k_per_wt: pd.DataFrame,
        save_dir: Optional[Path] = None,
        csv_name: str = "anom_overview.csv"
    ) -> pd.DataFrame:
        
        if per_event_df is None or per_event_df.empty:
            raise ValueError("per_event_df is empty.")
        if best_k_per_wt is None or best_k_per_wt.empty:
            raise ValueError("best_k_per_wt is empty.")
        
        per_event_df = per_event_df.copy()
        
        best = best_k_per_wt[[ic.WT_ID, "k"]].copy()
        
        merged = per_event_df.merge(best, on=[ic.WT_ID, "k"], how="inner")
        
        cols_order = [
            ic.WT_ID,
            "threshold",
            "sigma", 
            "k", 
            AnomOverviewKeys.ANOM_TYPE,
            AnomOverviewKeys.SIGNALS,
            AnomOverviewKeys.TS_START,
            AnomOverviewKeys.TS_END,
            AnomOverviewKeys.LATENCY,
            AnomOverviewKeys.INTENSITY,
        ]
        
        df = (
            merged[cols_order]
            .sort_values([ic.WT_ID, AnomOverviewKeys.TS_START])
            .reset_index(drop=True)
        )
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fp = Path(save_dir) / csv_name
            df.to_csv(fp, index=False)
        
        return df
        
    #########################################
    @classmethod
    def _k_grid(cls,
                max: float = 8,
                min: float = 1,
                step: float = 0.5
                ) -> List[float]:
        """ _k_grid will be used for the sigma thresholding.
            """
        n = int(round((max - min)/ step)) +1 # max-(n-1)*step = min
        return [max - i*step for i in range(n)]
    @classmethod
    def filter_by_thresh_and_postprocess(cls,
                                    signal_loss_df: pd.DataFrame,
                                    thresholds_df: pd.DataFrame,
                                    min_anom_len: int = 0,
                                    merge_gap: int = 1,
                                    ) -> pd.DataFrame:
        """ if loss >= threshold -> 1 else 0. \n
            Thresholds per WT,Signal pair in signal_loss_df \n
            Consecutive 1's will be aggregated to [ts_start, ts_end] \n
            Anomalies or outliers will be ignored if their length len <= min_anom_len \n
            Adjacent anomalies which are separated by a gap with length of merge_gap will be merged to one anomaly \n
            

        Args:
            signal_loss_df (pd.DataFrame): This DataFrame is obtained from prepare_df_eval_threshold_computing().\n 
            thresholds_df (pd.DataFrame): This DataFrame is obtained from threshold_from_mu_sigma(). \n
            min_anom_len (int, optional): The minimal length of an anomaly that will be registered. Defaults to 1. \n
            merge_gap (int, optional): Gap length of between 2 adjacent anomalies. Defaults to 1. \n

        Returns:
            pd.DataFrame: [anom_id, ic.WT_ID, ic.SIGNAL_COL, AnomOverviewKeys.TS_START, AnomOverviewKeys.TS_END]
        """
        
        df = signal_loss_df.merge(thresholds_df, on=[ic.WT_ID, ic.SIGNAL_COL], how="left")
        df = df.sort_values([ic.WT_ID, ic.SIGNAL_COL, ic.TS_COL])
        df["detected"] = (df[ic.RE_COL] >= df["threshold"]).astype(int)
        
        anom_rows = []
        #TODO
        # can we vectorize this loop
        for (wt, sig), group in df.groupby([ic.WT_ID, ic.SIGNAL_COL], sort=False):
            start_ts = None
            prev_ts = None
            
            for _, row in group.iterrows():
                
                if row["detected"] == 1 and start_ts is None:
                    start_ts = row[ic.TS_COL]
                if row["detected"] == 0 and start_ts is not None:
                    anom_rows.append([wt, sig, start_ts, prev_ts])
                    #reset
                    start_ts = None
                prev_ts = row[ic.TS_COL]
            
            if start_ts is not None:
                anom_rows.append([wt, sig, start_ts, prev_ts])
        
        anom_df = pd.DataFrame(anom_rows, columns=[ic.WT_ID, ic.SIGNAL_COL, AnomOverviewKeys.TS_START, AnomOverviewKeys.TS_END])
        
        if anom_df.empty:
            anom_df[AnomOverviewKeys.ANOM_ID] = pd.Series(dtype=int)
            return anom_df
        
        
        min_len = min_anom_len * cls.STEP
        anom_df = anom_df[(anom_df[AnomOverviewKeys.TS_END] - anom_df[AnomOverviewKeys.TS_START]) >= min_len].copy()
        anom_df = anom_df.sort_values([ic.WT_ID, ic.SIGNAL_COL, AnomOverviewKeys.TS_START])
        
        merged = []
        #TODO
        # can we vectorize this loop
        for (wt, sig), group in anom_df.groupby([ic.WT_ID, ic.SIGNAL_COL], sort=False):
            
            curr_start = None
            curr_end = None
           
            for _ , row in group.iterrows():
                
                if curr_start is None:
                    curr_start = row[AnomOverviewKeys.TS_START]
                    curr_end = row[AnomOverviewKeys.TS_END]
                elif row[AnomOverviewKeys.TS_START] - curr_end < merge_gap * cls.STEP:
                    curr_end = max(curr_end, row[AnomOverviewKeys.TS_END])
                else:
                    merged.append([wt, sig, curr_start, curr_end])
                    curr_start = row[AnomOverviewKeys.TS_START]
                    curr_end = row[AnomOverviewKeys.TS_END]
            
            if curr_start is not None:
                merged.append([wt, sig, curr_start, curr_end])
        
        anom_df = pd.DataFrame(merged, columns=[ic.WT_ID, ic.SIGNAL_COL, AnomOverviewKeys.TS_START, AnomOverviewKeys.TS_END])
        anom_df[AnomOverviewKeys.ANOM_ID]= range(len(anom_df))
        anom_df[ic.WT_ID] = anom_df[ic.WT_ID].astype(int)
        
        return anom_df
    
    @classmethod
    def _overlap_hit(cls,
                 gt_start: pd.Timestamp,
                 gt_end: pd.Timestamp,
                 det_start: pd.Timestamp,
                 det_end: pd.Timestamp
                 ) -> bool:
        if gt_start == gt_end:
            return (det_start <= gt_start) and (gt_start <= det_end)
        
        if det_start == det_end:
            return (gt_start <= det_start) and (det_start <= gt_end)
        
        return (min(gt_end, det_end) - max(gt_start, det_start)) > pd.Timedelta(0)
        
    @classmethod
    def corr_pairs_from_AnomalySpecs(cls, specs: List[AnomalySpec]) -> List[Tuple[int, str, str]]:
        corr_pairs : List[Tuple[int, str, str]] = []
        for spec in specs:
            if _cat_enum(spec.category) == AnomCategory.CORR:
                if not spec.secondary_signal:
                    raise ValueError(f"Corr spec for wt {spec.wt_id} need another secondary signal.")
                corr_pairs.append((spec.wt_id, spec.primary_signal, spec.secondary_signal))
        
        return corr_pairs
    
    @classmethod
    def _complement_intervalls(cls,
                               spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                               start: pd.Timestamp,
                               end: pd.Timestamp,
                               ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        
        if end <= start:
            return []
        
        spans = [(max(start, pd.to_datetime(a)), min(end, pd.to_datetime(b))) for (a,b) in spans]
        spans = [(a,b) for (a,b) in spans if a < b]
        spans.sort(key=lambda x: x[0])
        
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s,e in spans:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        
        complement: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        curr = start
        for s,e in merged:
            if curr < s:
                complement.append((curr,s))
            curr = max(curr, e)
        
        if curr < end:
            complement.append((curr, end))
        
        return complement
    
    @classmethod
    def top_corr_pairs_global(cls,
                              df_train: pd.DataFrame,
                              feature_cols: List[str],
                              top_n: int = 14,
                              
    )-> pd.DataFrame:
        """

        Returns:
            pd.DataFrame: ["s1", "s2", "rho", "rho_abs", "count"]
        """
        if len(feature_cols) == 0:
            raise ValueError(f"feature_cols is empty")
        
        X = df_train[feature_cols].copy()
        
        std = X.std(ddof=0)
        non_zero = std.index[std > 0]
        X = X[non_zero]
        
        Corr = X.corr(method="pearson")
        valid =(~X.isna()).astype(int)
        counts = valid.T @ valid
        
        corr_df = (
            Corr.where(np.triu(np.ones(Corr.shape, dtype=bool), k=1))
            .stack()
            .rename("rho")
            .reset_index()
            .rename(columns={"level_0": "s1", "level_1": "s2"})
        )
        
        counts_df = (
            pd.DataFrame(counts, index=Corr.index, columns=Corr.columns)
            .where(np.triu(np.ones(Corr.shape, dtype=bool),k=1))
            .stack()
            .rename("count")
            .reset_index(drop=True)
        )
        
        corr_df["count"] = counts_df.values
        
        corr_df = corr_df[corr_df["count"] >= 1000].copy()
        
        corr_df["rho_abs"] = corr_df["rho"].abs()
        corr_df = corr_df.sort_values("rho_abs", ascending=False)
        
        return corr_df.head(top_n).reset_index(drop=True)
    

    
class Eval_Anom:
    @staticmethod
    def quick_corr_stats(
        s1_scaled: np.ndarray,
        s2_scaled_old: np.ndarray, 
        s2_scaled_new: np.ndarray,
    )-> Dict:
        def corr(a,b):
            a = a - a.mean()
            b = b - b.mean()
            std_a = a.std(ddof=0)
            std_b = b.std(ddof=0)
            return float((a*b).mean() / (std_a * std_b + 1e-12))
        return {
            "rho_before": corr(s1_scaled, s2_scaled_old),
            "rho_after": corr(s1_scaled, s2_scaled_new),
            "mu_before": float(s2_scaled_old.mean()),
            "mu_after": float(s2_scaled_new.mean()),
            "std_before": float(s2_scaled_old.std(ddof=0)),
            "std_after": float(s2_scaled_new.std(ddof=0)),
            
        }
    
    @classmethod
    def build_threshold_table_part2(cls,
                                    mu_sigma_df: pd.DataFrame,
                                    best_k_per_wt: pd.DataFrame
                                    ) -> pd.DataFrame:
        """ Saves the Dataframe under ic.PATH_PART1_THRESHOLDS

        Returns:
            pd.DataFrame: columns [ic.WT_ID, ic.SIGNAL_COL, "threshold","mu", "sigma", "k"]
        """
        
        
        if mu_sigma_df.empty or best_k_per_wt.empty:
            raise ValueError(f"build_threshold_table_part2 parameters: mu_sigma_df or best_k_per_wt is empty")
        
        if "sigma" not in mu_sigma_df.columns:
            raise KeyError("sigma not in col of mu_sigma_df")
        
        if "mu" not in mu_sigma_df.columns:
            raise KeyError("The column mu is not in mu_sigma_df")
        
        merged = mu_sigma_df.merge(best_k_per_wt[[ic.WT_ID, "k"]], on=ic.WT_ID, how="left")
        merged["threshold"] = merged["mu"] + merged["k"] * merged["sigma"]
        
        cols = [ic.WT_ID, ic.SIGNAL_COL, "threshold","mu", "sigma", "k"]
        
        os.makedirs(ic.PATH_PART1_THRESHOLDS, exist_ok=True)
        
        df = merged[cols].sort_values([ic.WT_ID, ic.SIGNAL_COL]).reset_index(drop=True)
        
        df.to_csv(ic.PATH_PART1_THRESHOLDS / "thresholds.csv", index=False)
        
        return df
    
    @classmethod
    def build_diff_df_for_signal(cls,
        wt_id: int,
        signal: str,
        wt_df_raw : pd.DataFrame | Path= ic.PATH_IMPUTED,
        wt_df_inj : pd.DataFrame |Path = ic.PATH_PART1_VAL_SET_INJ_DIR,
        ts_range: Optional[Tuple[str,str]] = None,
    ) -> pd.DataFrame:
        
        if isinstance(wt_df_raw, str):
            raw_files = list(Path(wt_df_raw).glob(f"WT_ID_{wt_id}_*"))
            assert raw_files, f"No raw files for wt {wt_id} found."
            wt_df_raw: pd.DataFrame = pd.read_csv(raw_files[0])
        if isinstance(wt_df_inj, str):
            inj_files = list(Path(wt_df_inj).glob(f"wt_{wt_id}_*"))
            assert inj_files, f"No inj files for wt {wt_id} found"
            wt_df_inj: pd.DataFrame = pd.read_csv(inj_files[0])
        
        wt_df_raw[ic.TS_COL] = pd.to_datetime(wt_df_raw[ic.TS_COL])
        wt_df_inj[ic.TS_COL] = pd.to_datetime(wt_df_inj[ic.TS_COL])
        
        if ts_range is not None:
            ts_start, ts_end = ts_range
            ts_start = pd.to_datetime(ts_start)
            ts_end = pd.to_datetime(ts_end)
            
            mask_raw = (wt_df_raw[ic.TS_COL] >= ts_start) & (wt_df_raw[ic.TS_COL] <= ts_end)
            mask_inj = (wt_df_inj[ic.TS_COL] >= ts_start) & (wt_df_inj[ic.TS_COL] <= ts_end)
            
            wt_df_raw = wt_df_raw.loc[mask_raw].copy()
            wt_df_inj = wt_df_inj.loc[mask_inj].copy()
            
        sort_cols = [ic.TS_COL, ic.WT_ID]
        
        wt_df_raw = wt_df_raw.sort_values(sort_cols).reset_index(drop=True)
        wt_df_inj = wt_df_inj.sort_values(sort_cols).reset_index(drop=True)
        
        assert len(wt_df_raw) == len(wt_df_inj), "len of wt_df_raw is not equal with wt_df_inj"
        assert wt_df_raw[ic.TS_COL].equals(wt_df_inj[ic.TS_COL]), "Timestamps are not equal"
        assert wt_df_raw[ic.WT_ID].equals(wt_df_inj[ic.WT_ID]), "WT_ID columns are not equal"
        
        s_raw = pd.to_numeric(wt_df_raw[signal], errors="raise")
        s_inj = pd.to_numeric(wt_df_inj[signal], errors="raise")
        
        diff = (s_raw - s_inj).abs()
        
        df_diff = pd.DataFrame(
            {
                ic.TS_COL: wt_df_raw[ic.TS_COL],
                ic.WT_ID: wt_df_raw[ic.WT_ID],
                signal: diff,
            }
        )
        
        return df_diff
    
    @classmethod
    def _prep(cls,df: pd.DataFrame,wt_id:int, ts_col:str = ic.TS_COL, loss_col:str = ic.MEAN_LOSS_PER_SAMPLE) -> pd.DataFrame:
            x= df[[ts_col, ic.WT_ID, loss_col]].copy()
            x[ts_col] = pd.to_datetime(x[ts_col], errors="raise")
            x[ic.WT_ID] = pd.to_numeric(x[ic.WT_ID], errors="raise").astype(int)
            x[loss_col] = pd.to_numeric(x[loss_col], errors="raise").astype(float)
            if wt_id is not None:
                x = x[x[ic.WT_ID] == int(wt_id)].copy()
            return x
    
    @classmethod
    def _fmt_sci(cls,x: float, digits: int = 2) -> str:
            if not np.isfinite(x):
                return "-"
            return f"\\num{{{x:.{digits}e}}}"
    
    @classmethod
    def descr_stats_raw_inj(cls,
                            df_raw: pd.DataFrame,
                            df_inj: pd.DataFrame,
                            anom_spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                            anom_span_labels: List[str],
                            theta: float,
                            wt_id: int | None = None,
                            ts_col: str = ic.TS_COL,
                            loss_col: str = ic.MEAN_LOSS_PER_SAMPLE,
                            steps: pd.Timedelta = pd.Timedelta(minutes=10),
                            ) -> pd.DataFrame:
        if len(anom_spans) != len(anom_span_labels):
            raise ValueError("anom_spans and anom_span_labels have different lengths")
        
       
        
        # def _floor(x:float)-> float:
        #     return float(np.floor(x*10000.0) / 10000.0)
        
        # def _fmt_mstd(mu: float, sd: float) -> str:
        #     if not ((np.isfinite(mu)) and np.isfinite(sd)):
        #         return "-"
        #     return f"{mu:.4f} ± {sd:.4f} "
        
        # def _fmt_mstd_sci(mu: float, sd: float, digits: int=2) -> str:
        #     if not ((np.isfinite(mu)) and np.isfinite(sd)):
        #         return "-"
        #     return f"\\num{{{mu:.{digits}e}}} $\\pm$ \\num{{{sd:.{digits}e}}}"
        
        
        
        base = cls._prep(df_raw,wt_id=wt_id)
        inj = cls._prep(df_inj,wt_id=wt_id)
        
        rows = []
        for (s,e), label in zip(anom_spans, anom_span_labels):
            s = pd.to_datetime(s)
            e = pd.to_datetime(e)
            
            if "point" in label.lower() or (s == e):    
                b = base[(base[ts_col] >= s) & (base[ts_col] <= e)][[ts_col,loss_col]]
                inj_window = inj[(inj[ts_col] >= s) & (inj[ts_col] <= e)][[ts_col,loss_col]]
            else: 
                b = base[(base[ts_col] >= s) & (base[ts_col] < e)][[ts_col,loss_col]]
                inj_window = inj[(inj[ts_col] >= s) & (inj[ts_col] < e)][[ts_col,loss_col]]
            
            b_loss = b[loss_col]
            i = inj_window[loss_col]
            n = len(i)
            n_b = len(b_loss)
            det_base = (b_loss >= theta).sum()
            detected = (i >= theta).sum()
            cov_base = (det_base/ n_b) if n_b > 0 else np.nan
            coverage = (detected / n) if n > 0 else np.nan
            
            if detected > 0:
                t_first = inj_window.loc[inj_window[loss_col] >= theta, ts_col].iloc[0]
                latency = (t_first - s) / steps
            else:
                latency = np.nan
            
            if det_base > 0:
                t_first_b = b.loc[b[loss_col] >= theta, ts_col].iloc[0]
                latency_b = (t_first_b - s) / steps 
            else:
                latency_b = np.nan
                
            b_mean = b_loss.mean() if len(b_loss) > 0 else np.nan
            b_std =  b_loss.std(ddof=0) if len(b_loss) > 0 else np.nan
            i_mean = i.mean() if n > 0 else np.nan
            i_std =  i.std(ddof=0) if n > 0 else np.nan
            
            rows.append({
                "Anomaly Type": label,
                "BL μ": cls._fmt_sci(b_mean),
                "BL σ": cls._fmt_sci(b_std),
                "Anom. μ": cls._fmt_sci(i_mean),
                "Anom. σ": cls._fmt_sci(i_std),
                "BL Latency": latency_b,
                "Anom. Latency": latency,
                "BL Coverage": cov_base,
                "Anom. Coverage": coverage ,
                "Samples": n if "point" not in label.lower() else 1
            })
        
        df= pd.DataFrame(rows)
        
        return df
            
    @classmethod
    def best_signal_in_windows(cls,
                               df_re: pd.DataFrame,
                               windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
                               window_to_cat: Optional[Dict[Tuple[pd.Timestamp, pd.Timestamp], Any]],
                               window_cats: Optional[List[Any]] = None,
                               use_signals: Optional[List[str]] = None,
                               close_left: bool = True) -> pd.DataFrame:
        candidates = []
        use_ts_col = ic.TS_COL in df_re.columns
        if use_ts_col:
            ts = pd.to_datetime(df_re[ic.TS_COL], errors="raise")
        else: 
            ts = pd.to_datetime(df_re.index, errors="coerce")
                                
        for i, (start, end) in enumerate(windows):
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            
            cat = None
            if window_cats is not None:
                cat = window_cats[i]
            elif window_to_cat is not None:
                cat = window_to_cat.get((start,end), None)
            
            if start == end:
                mask = (ts == start)
            else:
                if close_left:
                    mask = (ts >= start) & (ts < end)
                else:
                    mask = (ts >= start) & (ts <= end)
                
            w = df_re.loc[mask]
            
            if w.empty:
                candidates.append({"Start" :start,
                                   "End": end,
                                   "Category": cat,
                                   "Signal": None,
                                   "Max RE": None})
                continue
            re_cols = [c for c in w.columns if isinstance(c,str) and c.startswith("RE_")]
            
            if use_signals is not None:
                allowed = set()
                for s in use_signals:
                    if s in w.columns:
                        allowed.add(s)
                    if isinstance(s, str) and ("RE_" + s) in w.columns:
                        allowed.add("RE_"+ s)
                re_cols = [c for c in re_cols if c in allowed]
                
            
            w = w[re_cols]
            
            max_per_sig = w.max()
            best_sig = max_per_sig.idxmax()
            
            candidates.append({
                "Start":start,
                "End": end,
                "Category": cat,
                "Signal": best_sig,
                "Max RE": max_per_sig[best_sig]
            })
        return pd.DataFrame(candidates)
        
class Inject_Anomalies:

    @classmethod
    def inject_anom_type(
        cls,
        df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        slope: float,
        target_col: str,
        operator: Anom_Type,
        wt_id: int,
        ts_col: str = ic.TS_COL,
        )-> pd.DataFrame:
        """ Imputes a anomaly type given by the class Anom_Type in a specified time period and column in the df table. \n
            More information about the operations of the anomaly type in the code below. (few lines)
            
            This method is used as a parameter for filter_fns in helperfunctions.helper.build_dataloaders()
            ------------------------------------------------------------------------------
        Args:
            df (pd.DataFrame): The anomalies will be imputed in df.
            start_ts (pd.Timestamp): The start of the imputation time period.
            end_ts (pd.Timestamp): The end of the imputation time period
            slope (float): The slope will be used to manipulate the values in df
            target_col (str): The column in df to manipulate
            operator (Anom_Type): Choose the anomaly type to impute
            ts_col (str, optional): Defaults to TS_COL.
            enforce_nonneg (bool, optional): To enforce non negative values. It might happen that the constructed signal has negative values.
            lower_bound (float, optional): Set the lower bound of the constructed signal.
            max_iters (int, optional): Used to renormalize the constructed signal until the signal is within tolerance (tol_corr).
            tol_corr (float, optional): The repeated renormalization of the constructed signal should be within a tolerance tol_corr.
            

        Returns:
            pd.DataFrame: The updated df containing a synthetic anomaly.
        """
        start_ts = pd.to_datetime(start_ts)
        end_ts = pd.to_datetime(end_ts)
        if (end_ts < start_ts):
            raise ValueError(f"start_ts cannot be after end_ts.")
        
        if not (target_col in df.columns):
            raise ValueError(f"target_col: {target_col} not in df table"
                            f"cols in df: {(print(c) for c in df.columns)}")
        if (isinstance(start_ts, str) or isinstance(end_ts, str)):
            start_ts = pd.to_datetime(start_ts)
            end_ts = pd.to_datetime(end_ts)
            
        elif not (isinstance(start_ts, pd.Timestamp) and (isinstance(end_ts, pd.Timestamp))):
            raise TypeError(f"start_ts is of type: {type(start_ts)}, and end_ts is of type: {type(end_ts)}"
                            f"start_ts: {start_ts} and end_ts: {end_ts}")
        
        df[ts_col] = pd.to_datetime(df[ts_col])
        mask_wt = (df[ic.WT_ID].astype(int) == int(wt_id))
        
        if not mask_wt.any():
            return df
        
        df_wt = df.loc[mask_wt, [ts_col, target_col]].copy()
        df_wt = df_wt.set_index(ts_col).sort_index()
        
        ts_unique = pd.Index(df_wt.index.unique().sort_values())
        
        idx_start = ts_unique.searchsorted(start_ts, side="left")
        idx_end = ts_unique.searchsorted(end_ts, side="right") -1
        
        if(idx_start >= len(ts_unique)) or (idx_end < 0):
            # raise ValueError(f"Could not find start- or end_ts in df"
            #                 f"start_ts:{start_ts}, end_ts:{end_ts}"
            #                 f"Smallest ts: {ts_unique.min()}, greatest ts: {ts_unique.max()}"
            #                 f"Is start_ts in df => {start_ts in ts_unique} "
            #                 f"Is end_ts in df => {end_ts in ts_unique} ")
            return df
        
        start = ts_unique[idx_start]
        end = ts_unique[idx_end]
        
        if (start_ts != start):
            print(f" start_ts:{start_ts} not found. start is now: {start}")
        if (end_ts != end):
            print(f" end_ts:{end_ts} not found. end is now: {end}")
            
        start_ts = start
        end_ts = end
        
        mask_time = (df[ts_col] >= start_ts) & (df[ts_col] <= end_ts)
        mask = mask_wt & mask_time
        if not mask.any():
            # raise ValueError(f"Error in wt_id:{wt_id} or time period start:{start_ts}, end:{end_ts}."
            #                  f"Check wt_id is in df = {str(wt_id) in df.columns}"
            #                  f"Check start_ts is in df = {pd.to_datetime(start_ts) in df.index}"
            #                  f"Check end_ts is in df = {pd.to_datetime(end_ts) in df.index}")
            return df

        match operator:
            case Anom_Type.CONST_OFFSET:
                df.loc[mask, target_col] = (
                    df.loc[mask, target_col].astype(float, copy=False) + slope
                )
            case Anom_Type.MULT_DRIFT:
                duration = (end_ts - start_ts)
                if duration <= pd.Timedelta(0):
                    tau = pd.Series(0.0, index=df.index)
                else:
                    tau = ((df.loc[mask, ts_col] - start_ts) / duration).clip(0.0,1.0)
                factor = 1.0 + slope * tau**2
                df.loc[mask, target_col] = (
                    df.loc[mask, target_col].astype(float, copy=False) * factor.to_numpy()
                )
            case _:
                raise ValueError(f"Operator unknown: {operator}. Known operations: {Anom_Type._member_names_}")

        return df
    
    @classmethod
    def _renorm_to_mu_sigma(cls,
                            x: np.ndarray, 
                            mu: float, 
                            sigma:float,
                            enforce_lower_bound: bool=True,
                            lower_bound: float = 0.0,
                            ) -> np.ndarray:
        x_mean = float(x.mean())
        x_std = float(x.std(ddof=0))
        
        if x_std == 0.0:
            eps = max(1e-10, 1e-7*sigma)
            det_ramp = np.arange(x.size, dtype=float)
            det_ramp -= det_ramp.mean()
            #add small variance
            x = x + eps*det_ramp
            x_mean = float(x.mean())
            x_std = float(x.std(ddof=0))
        
        if enforce_lower_bound:
            b0 = sigma /x_std if x_std > 0.0 else 1.0
            # min(a+b*x) >= lower_bound
            mmin = float(x.min()) 
            denom = (x_mean - mmin) + 1e-12
            bmax = (mu -lower_bound) / denom
            b = max(0.0, min(b0, bmax*0.999999))
            a = mu - b*x_mean
            y = a + b * x
            
            #numeric clip against machine error
            if lower_bound != -np.inf:
                y = np.maximum(y, lower_bound - 1e-12)
            
            return y
            
        
        b = sigma / x_std if x_std > 0.0 else 1.0 # 1.0: special case window with 1 ts, numeric underflow
        
        # numeric problem..
        if b < 0:
            b = -b
        a = mu - b* x_mean
        
        return a+b*x
    
    @classmethod
    def _correlation(cls,
                     a: np.ndarray,
                     b: np.ndarray,
                     ) -> float:
        a_centered = a-a.mean()
        b_centered = b-b.mean()
        a_std = float(a_centered.std(ddof=0))
        b_std = float(b_centered.std(ddof=0))
        if a_std == 0.0 or b_std == 0.0:
            return 0.0
        return float((a_centered*b_centered).mean() /  (a_std*b_std))
    
    @classmethod    
    def inject_corr_anomaly(
        cls,
        df: pd.DataFrame,
        signal1: str,
        signal2: str,
        window: Tuple[pd.Timestamp| str, pd.Timestamp |str],
        corr_target: float,
        wt_id: int,
        seed: int,
        ts_col: str = ic.TS_COL,
        edge_taper_frac: float = 0.0,
        enforce_nonneg: bool = True,
        lower_bound: float = 0.0,
        max_iters: int = 5,
        tol_corr: float = 1e-3
    )-> pd.DataFrame:
        
        if corr_target > 1 or corr_target < -1:
            raise ValueError(f"The correlation coefficient corr_target={corr_target} must be in [-1,1]")
        
        ts_start, ts_end = [pd.to_datetime(ts) for ts in window]
        
        
        if (signal1 not in df.columns) or (signal2 not in df.columns) or (ts_col not in df.columns):
            return df
        
        df = df.copy()
        
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
        
        mask_wt = (df[ic.WT_ID].astype(int) == int(wt_id))
        
        if not mask_wt.any():
            return df
        
        df_wt_ts = df.loc[mask_wt, [ts_col]].copy().set_index(ts_col).sort_index()
        ts_unique = pd.Index(df_wt_ts.index.unique()).sort_values()
        
        if ts_unique.empty:
            return df
        
        idx_start = ts_unique.searchsorted(ts_start, side="left")
        idx_end = ts_unique.searchsorted(ts_end, side="right") -1
        
        if (idx_start >= len(ts_unique)) or (idx_end < 0):
            return df
        
        start = ts_unique[idx_start]
        end = ts_unique[idx_end]
        
        mask_time = (df[ts_col] >= start) & (df[ts_col] <= end)
        mask = mask_wt & mask_time
        
        if not mask.any():
            return df
        
        #
        feature_cols = load_feature_order()
        
        assert signal1 in feature_cols, f"{signal1} not in feature_cols"
        assert signal2 in feature_cols, f"{signal2} not in feature_cols"
        
        s1 = df.loc[mask, signal1].astype(float).to_numpy(copy=False)
        s2 = df.loc[mask, signal2].astype(float).to_numpy(copy=False)
        
        mean_s1 = s1.mean()
        mean_s2 = s2.mean() 
        
        s1_centered = s1 - mean_s1
        s2_centered = s2 - mean_s2
        
        #biased (divide by n)
        sigma_s1 = s1.std(ddof=0) 
        sigma_s2 = s2.std(ddof=0) 
        
        #TODO
        # what if sigma_x is zero
        if sigma_s1 == 0.0 or sigma_s2 == 0.0:
            raise ValueError(f"sigma_s1:{sigma_s1}, sigma_s2:{sigma_s2} - constant signal")
        
        s1_zscored = s1_centered / sigma_s1
        #s2_zscored = s2_centered / sigma_s2
        
        var_s1 = np.dot(s1_centered, s1_centered)
        cov_s1s2 = np.dot(s1_centered, s2_centered)
        
        alpha = cov_s1s2 / var_s1 if var_s1 != 0 else 0.0
        
        # correlation coefficient p(s1,s2)
        n = s1_centered.size
        rho_s1s2 = (cov_s1s2/ n )/ ( sigma_s1 * sigma_s2)
        
        # consistency check
        if not np.allclose(alpha, rho_s1s2 * (sigma_s2/sigma_s1)) == True:
            raise ValueError(f"alpha = cov_s1s2 / var_s1 != rho_s1s2 * (sigma_s2/sigma_s1)"
                             f"alpha= {alpha} != {rho_s1s2 * (sigma_s2/sigma_s1)}")
            
        
        epsilon = s2_centered - alpha*s1_centered
        epsilon_centered = epsilon - epsilon.mean()
        sigma_epsilon = epsilon_centered.std(ddof=0)
        
        # if s1 and s2 are perfectly linear dependent
        if np.allclose(sigma_epsilon, 0.0):
            rng = np.random.default_rng(seed)
            
            # the samples in noise are not exactly ~ N(0,1) !!
            noise = rng.standard_normal(size=s1_centered.shape)
            
            
            # orthogonalization against s2_centered (Gram-Schmidt)
            proj = (np.dot(s1_centered, noise)) / (np.dot(s1_centered, s1_centered)) if np.dot(s1_centered, s1_centered) != 0 else 0.0
            noise_perpendicular = noise - proj * s1_centered
            noise_perpendicular -= noise_perpendicular.mean()
            sigma_noise_perp = noise_perpendicular.std(ddof=0)
            #TODO
            # it might be possible that sigma_noise_perp equals zero...
            if sigma_noise_perp == 0.0:
                raise ValueError(f"sigma_noise_perp is zero. Code needs to be corrected, e.g. create new noise ...")
            noise_perpendicular /= sigma_noise_perp
            
            #scaling
            factor = max(1e-7*sigma_s1, 1e-10)
            epsilon_centered = factor* noise_perpendicular
            sigma_epsilon = epsilon_centered.std(ddof=0)
            
        #z-scoring of s1 and epsilon
        s = s1_zscored
        t = epsilon_centered / sigma_epsilon
        
        # construct new signal s2' with Var(s2) = Var(s2'), Exp(s2) = Exp(s2')
        # with free parameter corr_target
        new_s2 = mean_s2 + sigma_s2 * (corr_target*s + np.sqrt(max(0.0, 1-corr_target**2))*t)
        
        #check
        mean_new_s2 = new_s2.mean()
        sigma_new_s2 = new_s2.std(ddof=0)
        assert np.allclose(mean_new_s2, mean_s2), \
            f"Exp(new_s2)={mean_new_s2} != mean_s2={mean_s2}"
        assert np.allclose(sigma_new_s2, sigma_s2), \
            f"sigma_new_s2={sigma_new_s2} != sigma_s2={sigma_s2}"
        
        if enforce_nonneg:
            rho_inner = float(corr_target)
            new_s2_adjusted = new_s2.copy()
            
            for _ in range(max_iters):
                new_s2_inner = mean_s2 + sigma_s2 * (rho_inner * s + np.sqrt(max(0.0, 1.0-rho_inner**2 ))*t)
                tmp = np.maximum(new_s2_inner, lower_bound)
                new_s2_adjusted = cls._renorm_to_mu_sigma(tmp, 
                                                          mean_s2, 
                                                          sigma_s2,
                                                          enforce_lower_bound=True,
                                                          lower_bound=lower_bound)
                
                rho_observed = cls._correlation(new_s2_adjusted, s1)
                
                if abs(rho_observed - corr_target) <= tol_corr:
                    break
                if abs(rho_observed) > 1e-12: # 
                    rho_inner = float(np.clip(rho_inner*(corr_target / rho_observed), -0.999999, 0.999999 ))
                else:
                    rho_inner = float(np.clip(0.5 * (rho_inner + corr_target), -0.999999, 0.999999))
            new_s2 = new_s2_adjusted
        
        # # s2 and new_s2 have not any more same variance. var(s2) = k^2 var(new_s2) with a small k
        # # this is needed otherwise the injected anomaly may have negative values in the raw data space.
        # new_s2 = cls._rescale_to_0_1_space(new_s2, mu2=float(new_s2.mean()), eps=0.01)
        
        # create smooth edges in the anomaly window
        if edge_taper_frac > 0.0:
            # window length
            n = s2.shape[0]
            # number of ts to manipulate on one side of the signal
            i = int(np.floor(edge_taper_frac*n))
            
            if i > 0 and i < n//2:
                taper = np.ones(n, dtype=float)
                
                t_taper = np.linspace(0, np.pi/2, i)
                taper[:i] =  np.cos(t_taper)
                taper[-i:] = 1-np.cos(t_taper[::-1])
                
                new_s2 = taper*s2 + (1-taper)*new_s2
        
        df.loc[mask, signal2] = new_s2.astype(float, copy=False)
        
        return df
        

class Report:
    
    @staticmethod
    def create_mu_sigma_ratio_table(
        df_base_eval: pd.DataFrame,
        df_inj_eval: pd.DataFrame,
        signals: list[str],
        anom_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
        anom_cats: list[str],
        wt_id: int = 1,
    ) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        
        if len(anom_spans) != len(anom_cats):
            raise ValueError(f"anom_cats and anom_spans do not fit together.")
        base = df_base_eval.copy()
        inj = df_inj_eval.copy()
        
        base[ic.TS_COL] = pd.to_datetime(base[ic.TS_COL])
        inj[ic.TS_COL] = pd.to_datetime(inj[ic.TS_COL])
        
        base = base[base[ic.WT_ID] == wt_id]
        inj = inj[inj[ic.WT_ID] == wt_id]
        
        eps = 1e-10
        rows=[]
        for sig in signals:
            if sig != ic.MEAN_LOSS_PER_SAMPLE:
                col = f"{ic.RE_PREFIX}{sig}"
            else:
                col = sig
            row = {"Signal": sig}
            for idx, (s,e)in enumerate(anom_spans, start=1):
                s = pd.to_datetime(s)
                e = pd.to_datetime(e)
                if s == e:
                    mu_mask_b = (base[ic.TS_COL] == s)
                    mu_mask_i = (inj[ic.TS_COL] == s)
                
                else:
                    mu_mask_b = (base[ic.TS_COL] >= s) & (base[ic.TS_COL] < e)
                    mu_mask_i = (inj[ic.TS_COL] >= s) & (inj[ic.TS_COL] < e)

                sig_b = base.loc[mu_mask_b, col].astype(float).to_numpy()
                sig_i = inj.loc[mu_mask_i, col].astype(float).to_numpy()
                
                row[fr"{idx}. RCM"] = (sig_i.mean() / (sig_b.mean() + eps)) if (sig_i.size and sig_b.size) else np.nan
                row[fr"{idx}. RCSD"] = (sig_i.std(ddof=0) / (sig_b.std(ddof=0) + eps)) if (sig_i.size and sig_b.size) else np.nan
                sigma = (sig_b.std(ddof=0) + eps) if sig_b.size else np.nan
                mu = sig_b.mean() if sig_b.size else np.nan
                row[fr"{idx}. $\hat\sigma_{{base}}$"] = sigma
                row[fr"{idx}. $\hat\mu_{{base}}$"] = mu
            rows.append(row)

        df = pd.DataFrame(rows)
        df.at[0, "Signal"] = "MSE"
        mr_cols = [fr"{idx}. RCM" for idx in range(1, len(anom_spans)+1)]
        sr_cols = [fr"{idx}. RCSD" for idx in range(1, len(anom_spans)+1)]
        sigma_b_cols = [fr"{idx}. $\hat\sigma_{{base}}$" for idx in range(1, len(anom_spans)+1)]
        mu_b_cols = [fr"{idx}. $\hat\mu_{{base}}$" for idx in range(1,len(anom_spans)+1)]

        return df[["Signal",*mr_cols, *sr_cols]].round(3),df[["Signal", *mu_b_cols]].round(3), df[["Signal", *sigma_b_cols]].round(3), 
    

        
        
        
        
    