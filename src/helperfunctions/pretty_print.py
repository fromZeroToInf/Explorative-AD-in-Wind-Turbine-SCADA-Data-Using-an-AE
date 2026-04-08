import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mpdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import re
import pandas as pd
from helperfunctions.intern_constants import MEAN_LOSS_PER_SAMPLE, TS_COL, WT_ID, DIRECTORY_PRINTS, PATH_IMPUT_MASKS, ANY, RE_PREFIX
from helperfunctions import training_lib as tl
from helperfunctions.training_lib import HistoryDict
from pathlib import Path
import os
from typing import Optional, Callable, List, Tuple, Iterable, Any
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import seaborn as sns

from helperfunctions import intern_constants as ic

class PrettyPrint():
    
    @classmethod
    def find_wt_file(cls, files:list[str], wt_id:int|str)-> int| None:
                substring = f"WT_ID_{wt_id}_"
                for i, fp in enumerate(files):
                    if substring in fp:
                        return i
    
    #TODO need to be tested
    @classmethod
    def _merge_spans(cls,
                     spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                     ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """ Interval Union
        """
        if not spans:
            return []
        
        spans_sorted = sorted(spans, key=lambda x: x[0]) # x[0]= starts, x[1]= ends
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        
        cur_start, cur_end = spans_sorted[0] # earliest intervall
        
        for start, end in spans_sorted[1:]:
            if start <= cur_end: # overlapping or directly after
                if end > cur_end: # extend end
                    cur_end = end
            else: # no overlapping or directly following
                merged.append((cur_start, cur_end)) # close interval block
                cur_start, cur_end = start, end # start new merge block
        merged.append((cur_start, cur_end)) # last interval block
        
        return merged
    
    
    @classmethod
    def _clip_spans_to_range(cls,
                             spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                             t_min: pd.Timestamp,
                             t_max: pd.Timestamp,
                             ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        clipped = []
        for start, end in spans:
            if end <= t_min or start >= t_max:
                continue
            ss = max(start, t_min)
            ee = min(end, t_max)
            
            if ss <= ee:
                clipped.append((ss,ee))
        
        return clipped
    
    # @classmethod
    # def _load_impute_log(cls,file_path:str) -> pd.DataFrame:
    #     file_path = Path(file_path)
    #     df:pd.DataFrame = None
    #     if file_path.suffix.lower() == ".parquet":
    #         df = pd.read_parquet(file_path)
    #     elif file_path.suffix.lower() == ".csv":
    #         df = pd.read_csv(file_path)
        
    #     df[TS_COL] = pd.to_datetime(df[TS_COL])
    #     df["imputed"] = df["imputed"].astype(bool)
        
    #     return df
    
    @classmethod
    def _bool_runs_to_spans_fixed_step(
        cls,
        timestamps: pd.DatetimeIndex | pd.Series,
        mask: pd.Series | np.ndarray,
        step: pd.Timedelta = pd.Timedelta(minutes=10),
        inclusive_end: bool = True
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """ The boolean series (true = gap) will be changed to [start, end] intervals. 
        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: [start, end] intervals
        """
        idx: pd.DatetimeIndex = None
        if isinstance(timestamps, pd.DatetimeIndex):
            idx = timestamps
        # fix
        elif isinstance(timestamps, pd.Series):
            idx = pd.to_datetime(timestamps)
        else:
            raise TypeError(f"timestamps is not of type pd.DatetimeIndex or pd.Series. Type of timestamps: {type(timestamps)}")

        mask = pd.Series(mask, index=idx, dtype=bool).sort_index()
        
        if mask.empty or not mask.any():
            return []
         
        start_flags = mask & ~ mask.shift(1, fill_value=False) # shift(1) downwards, gaps at the edges will be filled with False
        end_flags = mask & ~ mask.shift(-1, fill_value=False) # shift(-1) upwards, gaps at the edges will be filled with False
        
        starts = mask.index[start_flags]
        ends = mask.index[end_flags]
        
        # error handling
        n = min(len(starts), len(ends))
        
        spans: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, end in zip(starts[:n], ends[:n]):
            end2 = (end + step) if inclusive_end else end
            spans.append((pd.Timestamp(start), pd.Timestamp(end2)))
            
        return spans

    @classmethod
    def _add_spans(cls,
                   ax: plt.Axes,
                   spans: Iterable[Tuple[pd.Timestamp, pd.Timestamp]],
                   label: str = "NaN imputation",
                   alpha: float = 0.2,
                   zorder: int = 0.5,
                   add_legend = False,
                   color: Optional[str] = None,
                ) -> None:
        first = True
       
        for start, end in spans:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
           
            if pd.isna(start) or pd.isna(end):
                continue
            
            if end < start:
                raise ValueError(f"start:{start} is after end:{end}")
            
            span_kwargs = {}
            if color is not None:
                span_kwargs["color"] = color
            
            if start == end:
                ax.axvline(
                    start,
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=zorder,
                    label = label if add_legend and first else None,
                    **span_kwargs
                    )
            else:
                ax.axvspan(
                    start,
                    end,
                    alpha=alpha,
                    zorder=zorder,
                    label=label if add_legend and first else None,
                    **span_kwargs,
                )
            
            if first and add_legend:
                first = False
            
               
        
    # @classmethod
    # def load_anomaly_spans(cls,
    #                        anom_windows_path: Path,
    #                        wt_ids: Optional[List[int]] = None,
    #                        categories: Optional[Iterable[Union[ce.AnomCategory, str]]]= None, 
    #                        signals: Optional[List[str]]= None
    # ) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], pd.DataFrame]:
    #     df = pd.read_csv(
    #         anom_windows_path,
    #         parse_dates=[str(ce.AnomOverviewKeys.TS_START), str(ce.AnomOverviewKeys.TS_END)]
    #     )
        
    #     if wt_ids is not None:
    #         df = [df[df[ce.AnomOverviewKeys.WT_ID].astype(int).isin(list(map(int, wt_ids)))]]
            
    #     if categories is not None:
    #         cats = [(c.name if isinstance(c, ce.AnomCategory) else str(c)) for c in categories]
    #         df = df[df[ce.AnomOverviewKeys.CATEGORY].isin(cats)]
            
    #     if signals is not None:
    #         sigs = set(map(str, signals))
    #         df = df[df[ce.AnomOverviewKeys.SIGNALS].astype(str).isin(sigs)]
        
    #     spans = list(zip(df[ce.AnomOverviewKeys.TS_START], df[ce.AnomOverviewKeys.TS_END]))
        
    #     return spans, df.reset_index(drop=True)
    
    
    @classmethod
    def print_loss(cls,
                    df:pd.DataFrame, 
                    save_filename: Optional[Path]=None,
                    ts_col:str= TS_COL, 
                    values:str = MEAN_LOSS_PER_SAMPLE,
                    dpi: int = 240,
                    title:str = "Mean Turbine Loss per Sample",
                    wt_id: Optional[List[int]] = None,
                    y_limits: Optional[Tuple[Tuple[float,float],Tuple[float,float]]]= None,
                    upper_ylim_to_max: bool = True,
                    #impute_log_directory: Optional[str] = None,
                    show_impute_t_periods: bool =True,
                    impute_label: str = "Data Imputation (Preprocessing)",
                    ts_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                    anom_span_label = "Injected Anomaly Windows",
                    anomaly_spans: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
                    line_width: float = 0.8,
                    legend_col: int = 4,
                    mark_threshold: Optional[float] = None,
                    detection_ts: Optional[List[pd.Timestamp]] = None,
                    detection_label: str = "Selected Detections",
                    marker: Optional[pd.Timestamp] = None,
                    marker_label: str = "Maintenance",
                    y_label: str = "Loss",
                    x_major_locator: Optional[Any] = None,
                    x_minor_locator: Optional[Any] = None,
                    x_data_format: Optional[str] = None,
                    show_anom_xticks: bool = True,
                    anom_span_labels: Optional[List[str]] = None,
                    figsize:Tuple[int,int]=(16,7),
                    add_wind_vs_nacelle: bool = False,
                    wind_col: str = "Wind direction (°)",
                    nacelle_col: str = "Nacelle position (°)",
                    power_col_wind_panel: str = "Power (kW)",
                    wind_panel_title: str = "Wind Direction vs. Nacelle Position",
                    show_mean: bool = True,
                    ):
        """ Plots the losses/signal values from df."""
        
        if (y_limits is not None) and add_wind_vs_nacelle:
            raise ValueError("add_wind_vs_nacelle=True is not supported with y_limits")
        
        missing = [col for col in (ts_col,WT_ID,values) if col not in df.columns]
        if len(missing) != 0:
             raise ValueError(f"Missing columns in df: {missing}."
                             f"Available columns:{df.columns}")
             
        if (save_filename is not None):
            os.makedirs(ic.PATH_PRINTS, exist_ok=True )
        
        df_cpy = df.copy()
        df_cpy[ts_col] = pd.to_datetime(df_cpy[ts_col]) 
        
        if ts_range is not None:
            if len(ts_range)!= 2:
                raise ValueError("ts_range must be tuple (start, end).")
            start, end = ts_range
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            
            if (start > end):
                raise ValueError(f"start:{start} cannot be after end:{end}")
            
            df_cpy = df_cpy[(df_cpy[ts_col] >= start) & (df_cpy[ts_col] <= end)]
            
            if df_cpy.empty:
                raise ValueError(f" No data in the selected time period ({start},{end})")

        df_out: pd.DataFrame = None
        if wt_id is not None:
            df_cpy = df_cpy.set_index(WT_ID, drop=False)
            df_out = df_cpy.loc[wt_id]
            df_out = df_out.reset_index(drop=True)
        
        wide = (
            df_cpy.pivot_table(index=ts_col,
                            columns=WT_ID,
                            values=values,
                            #aggfunc="mean"
                            )
            .sort_index() 
            
            if wt_id is None else (
                
            df_out.pivot_table(index=ts_col,
                            columns=WT_ID,
                            values=values
                            )
            .sort_index()
            )
        )
        wt_ids = list(wide.columns)
        
        if ts_range is not None:
            x_min = pd.to_datetime(ts_range[0])
            x_max = pd.to_datetime(ts_range[1])
        else:
            x_min = pd.to_datetime(wide.index.min())
            x_max = pd.to_datetime(wide.index.max())
        
        anomaly_spans_clipped: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        
        if anomaly_spans:
            anomaly_spans_clipped = cls._clip_spans_to_range(anomaly_spans, x_min, x_max)
        
        span_stats_list:List[Tuple[pd.Timestamp, pd.Timestamp, float, float]] = []
        if anomaly_spans_clipped:
            for (s0,e0) in anomaly_spans_clipped:
                s0 = pd.to_datetime(s0)
                e0 = pd.to_datetime(e0)
                
                m_span = (wide.index >= s0) & (wide.index < e0)
                if not m_span.any():
                    continue
                vals = wide.loc[m_span].to_numpy(dtype=float).ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals, ddof=0))
                span_stats_list.append((s0,e0,mu,sigma))
            
        
        anomaly_spans_with_labels: List[Tuple[Tuple[pd.Timestamp, pd.Timestamp], str]] = []
        if (anomaly_spans is not None) and (anom_span_labels is not None):
            if len(anom_span_labels) != len(anomaly_spans):
                raise ValueError("anom_span_labels must have the same length as anomaly_spans")

            for (s,e), label in zip(anomaly_spans, anom_span_labels):
                clipped = cls._clip_spans_to_range([(s,e)], x_min, x_max)
                if clipped:
                    anomaly_spans_with_labels.append((clipped[0], label))
            
            anomaly_spans_with_labels.sort(key=lambda x: x[0][0])
        
        detection_ts_clipped: List[pd.Timestamp] = []
        if detection_ts is not None:
            detection_ts_clipped = [
                pd.to_datetime(t) for t in detection_ts
                if (pd.to_datetime(t) >= x_min) and (pd.to_datetime(t) <= x_max)
            ]
        
        
        cmap = plt.get_cmap("tab20", len(wt_ids))
        color_map = {wt: cmap(i) for i, wt in enumerate(wt_ids)}
        
        if y_limits:
            
            fig, axes = plt.subplots(
                2,1, 
                sharex=True, 
                figsize=figsize,
                gridspec_kw={"hspace": 0.05}     
            )
            low, high = sorted(y_limits, key=lambda r: min(r))
            # bugfix
            low = (min(low), max(low))
            high = (min(high), max(high))
            if low[1] > high[0]:
                mid = (low[1]+ high[0]) / 2.0
                low = (low[0], mid)
                high = (mid, high[1])
                
                
            axes[0].set_ylim(*high)
            axes[1].set_ylim(*low)
            
            if upper_ylim_to_max:
                data_max = np.nanmax(wide.to_numpy(dtype=float))
                ymin, _ = axes[0].get_ylim()
                axes[0].set_ylim(ymin, data_max+ data_max*0.25)
            
            axes[0].spines["bottom"].set_visible(False)
            axes[1].spines["top"].set_visible(False)
            
            axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            
            d = .004
            axes[0].plot((-d,+d), (-d,+d), transform=axes[0].transAxes,
                         color="k", linewidth=1, clip_on=False)
            axes[1].plot((-d,+d), (1-d,1+d), transform=axes[1].transAxes,
                         color="k", linewidth=1, clip_on=False)
        else:
            if add_wind_vs_nacelle:
                fig, axes = plt.subplots(2,1, sharex=True, figsize=figsize, gridspec_kw={"hspace":0.2})
                
            else:
                fig, ax = plt.subplots(figsize=figsize)
                axes =[ax]
            #fix
            data_max = np.nanmax(wide.to_numpy(dtype=float))
            ymin, _ = axes[0].get_ylim()
            axes[0].set_ylim(ymin, data_max+ data_max*0.25)
        
        # Marking time periods of data imputation
        merged_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        
        if show_impute_t_periods:
            
            use_any = (wt_id is None) or (values == MEAN_LOSS_PER_SAMPLE)
            
            signal_key = ANY if use_any else values
            
            files = glob(os.path.join(PATH_IMPUT_MASKS, "*.parquet"))
            
            idx_plot = pd.to_datetime(wide.index)
            
            spans_per_wt: dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
            
            if use_any:
                cand = [ANY]
            else:
                cand = [signal_key]
                if signal_key.startswith(f"{RE_PREFIX}"):
                    cand.append(signal_key[3:])
            
            # fix
            if wt_id is None:
                wt_id = [1,2,4,5,6,7,8,9,10,11,12,13,14,15]
                
            for wt in wt_id:
                file_idx = cls.find_wt_file(files, wt)
                
                if file_idx is None:
                    raise ValueError(f"parquet file with wt_id: {wt} not found in {PATH_IMPUT_MASKS}")
                
                df_impute_log = pd.read_parquet(files[file_idx])
                df_impute_log[TS_COL] = pd.to_datetime(df_impute_log[TS_COL], errors="coerce")
                
                
                
                df_impute_log = df_impute_log[
                    (df_impute_log[WT_ID] == int(wt)) &
                    (df_impute_log[TS_COL] >= x_min) & (df_impute_log[TS_COL] <= x_max) &
                    (df_impute_log["Signal"].isin(cand))
                ]
                if df_impute_log.empty:
                    spans_per_wt[wt] = []
                    continue
                
                ts_events =pd.to_datetime(df_impute_log[TS_COL]).values
                mask = np.isin(idx_plot.values, ts_events)
                
                spans_per_wt[wt] = cls._bool_runs_to_spans_fixed_step(
                    idx_plot,
                    mask,
                    step=pd.Timedelta(minutes=10),
                    inclusive_end=True
                )
                
            all_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for wt in wt_ids:
                all_spans.extend(spans_per_wt.get(wt))
            
            all_spans = cls._clip_spans_to_range(all_spans, x_min, x_max)
            merged_spans = cls._merge_spans(all_spans)
        
        
        loss_axes = axes[:1] if add_wind_vs_nacelle else axes
            
        for ax_i, ax in enumerate(loss_axes):
            if merged_spans:
                cls._add_spans(
                    ax, 
                    merged_spans,
                    label=impute_label,
                    alpha=0.30,
                    zorder=0.4,
                    add_legend=(ax_i == 0),
                    color="red",
                )
            
            if anomaly_spans_clipped:
                cls._add_spans(ax,
                               anomaly_spans_clipped,
                               label=anom_span_label,
                               alpha=0.30,
                               zorder=0.8,
                               add_legend=(ax_i== 0),
                               color="green",
                               )
                if span_stats_list and show_mean:
                    for j, (s0,e0,mu,sigma) in enumerate(span_stats_list):
                        ax.hlines(
                            y=mu,
                            xmin=s0,
                            xmax=e0,
                            color="black",
                            linestyle="-",
                            linewidth=1.2,
                            alpha=0.95,
                            zorder=9,
                            label=f"Mean={mu:.3f}" if (ax_i == 0 and j ==0) else "_nolegend_"
                        )
                    ax.fill_between(
                        [s0,e0],
                        [mu - sigma, mu -sigma],
                        [mu + sigma, mu + sigma],
                        alpha=0.15,
                        zorder=8,
                        label=rf"$\mu={mu:.3f} \pm \sigma={sigma:.3f}$" if (ax_i == 0 and j==0) else "_nolegend_",
                    )
                    
                if (ax_i == 0) and anomaly_spans_with_labels:
                    y0, y1 = ax.get_ylim()
                    y_text = y1 - (y1 - y0)*0.3
                    for (s,e), label in anomaly_spans_with_labels:
                        if not label:
                            continue
                        
                        x_mid = s+(e-s)/2
                        ax.text(
                            x_mid, 
                            y_text, 
                            label,
                            ha="center",
                            va="top",
                            rotation = 90,
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.3),
                            zorder=20,
                            clip_on=True,
                        )
            
            if detection_ts_clipped:
                for j, t in enumerate(detection_ts_clipped):
                    ax.axvline(
                        t,
                        color="orange",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=10,
                        label=(detection_label if (ax_i==0 and j==0) else "_nolegend_"),
                    )
            if marker is not None:
                ax.axvline(marker,
                           color="red",
                           linestyle="--",
                           linewidth=1.2,
                           alpha=0.9,
                           zorder=10,
                           label=(marker_label if (ax_i==0 and j==0) else "_nolegend_"),
                           )
            for i, wt in enumerate(wide.columns):    
                ax.plot(wide.index, wide[wt], label=f"WT {wt}", linewidth=line_width, color=color_map[wt])
            ax.grid(True, alpha=0.3)
            
            if mark_threshold is not None:
                if ax_i == 0:
                    thr_label = f"Threshold = {mark_threshold:.3f}"
                else:
                    thr_label = "_nolegend_"
                
                ax.axhline(mark_threshold, ls="--", alpha=0.75, color="red", label=thr_label)
                # if mark_exceed:
                #     V = pd.DataFrame(wide, copy=False)
                #     m = (V.ge(mark_threshold)).any(axis=1).to_numpy()
                #     if m.any():
                #         ax.scatter(V.index[m], V[m].max(axis=1).to_numpy(), s=8, zorder=5, color="red")
            
        if add_wind_vs_nacelle:
            ax_ctx = axes[1]
            
            cls.plot_wind_vs_nacelle(df=df,
                                     ts_col=ts_col,
                                     wind_col=wind_col,
                                     nacelle_col=nacelle_col,
                                     ts_mark=(detection_ts_clipped[0] if detection_ts_clipped else None),
                                     ts_range=(x_min, x_max),
                                     title=wind_panel_title,
                                     power_col=power_col_wind_panel,
                                     power_label=power_col_wind_panel,
                                     save_filename=None,
                                     dpi=dpi,
                                     ax=ax_ctx,
                                     show=False,)
        axes[0].set_title(title)
        axes[-1].set_xlabel("Time")
        
        if y_limits is None:
            axes[0].set_ylabel(y_label)
            axes[0].yaxis.set_label_coords(-0.10, 0.5)
        else:
            axes[0].set_ylabel("")
            fig.text(0.02, 0.5, y_label, va="center", rotation="vertical")
        
        for ax_ in axes:
            ax_.margins(y=0.05)
        fig.subplots_adjust(left=0.10)
        
        leg = axes[0].legend(
            ncol=legend_col,
            fontsize=10,
            frameon=True,
            loc="upper right",
            framealpha=1.0,
            facecolor="white",
        )
        leg.set_zorder(1000)
        leg.set_clip_on(False)
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.3)
        frame.set_edgecolor("none")
        
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        
        if x_major_locator is None:
            x_major_locator = mpdates.AutoDateLocator(minticks=2, maxticks=10)
        if x_data_format is None:
            x_data_format = "%Y-%m-%d %H:%M"
        
        for ax_ in axes:
            ax_.set_xlim(x_min, x_max)
            ax_.xaxis.set_major_locator(x_major_locator)
            
            if(x_data_format is None) or (x_data_format == "concise"):
                locator = ax_.xaxis.get_major_locator()
                ax_.xaxis.set_major_formatter(mpdates.ConciseDateFormatter(locator))
            else:
                ax_.xaxis.set_major_formatter(mpdates.DateFormatter(x_data_format))
            
            if x_minor_locator is not None:
                ax_.xaxis.set_minor_locator(x_minor_locator)
            
            if show_anom_xticks and anomaly_spans_clipped and (ax_ is axes[-1]):
                ticks_dt = [pd.to_datetime(x_min), pd.to_datetime(x_max)]
                for s,e in anomaly_spans_clipped:
                    s = pd.to_datetime(s)
                    e = pd.to_datetime(e)
                    
                    if x_min <= s <= x_max:
                        ticks_dt.append(s)
                    if x_min <= e <= x_max:
                        ticks_dt.append(e)
                
                for t in detection_ts_clipped:
                    t = pd.to_datetime(t)
                    if x_min <= t <= x_max:
                        ticks_dt.append(t)
                
                if marker is not None:
                    marker_ts = pd.to_datetime(marker)
                    if x_min <= marker_ts <= x_max:
                        ticks_dt.append(marker_ts)
                
                ticks_num = sorted(set(mpdates.date2num(ticks_dt)))
                
                ax_.xaxis.set_major_locator(ticker.FixedLocator(ticks_num))
                ax_.xaxis.set_major_formatter(mpdates.DateFormatter("%Y-%m-%d %H:%M"))
                ax_.xaxis.set_minor_locator(ticker.NullLocator())
                ax_.xaxis.set_minor_formatter(ticker.NullFormatter())
                
                
                # if ax_ is axes[-1]:
                ax_.tick_params(
                    axis="x",
                    which="major",
                    length=6,
                    width=1,
                    pad=10,
                    labelbottom=True,
                    labelrotation=45,
                    labelsize=10,
                )
                # else:
                #     ax_.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
                #     ax_.xaxis.get_offset_text().set_visible(False)
        
        
        fig.autofmt_xdate()
        for ax_ in axes:
            ax_.tick_params(axis="x", which="both", labelrotation=30)
            for lbl in ax_.get_xticklabels(which="both"):
                lbl.set_ha("right")
                lbl.set_rotation_mode("anchor")
        
        if (save_filename is not None):
            fig.savefig(
                ic.PATH_PRINTS / save_filename,
                dpi=dpi,
                bbox_inches="tight"
            )
        plt.show()

    @staticmethod
    def print_learning_curve(
                            history: List[HistoryDict],
                            save_dir: Optional[Path] = None,
                            dpi:int = 240,
    ):
        if (save_dir is not None):
            target_path = (save_dir / DIRECTORY_PRINTS)
            os.makedirs(target_path, exist_ok=True)
        
        epochs = [h["epoch"] for h in history]
        train_re = [float(h["train_mean_epoch"]) for h in history]
        val_re = [float(h["val_mean_epoch"]) for h in history]

        plt.figure(figsize=(7,5))
        plt.plot(epochs, train_re, marker="o", linewidth=1.0, label="Train RE")
        plt.plot(epochs, val_re, marker="o", linewidth=1.0, label="Val RE")
        ax = plt.gca()
        
        n = len(epochs)
        
        step = max(1, n // 12)
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        
        ax.tick_params(axis="x", labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_horizontalalignment("right")
        ax.margins(x=0.01)
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if (save_dir is not None):
            plt.savefig(target_path / "Learning_Curve.png", 
                        dpi=dpi, 
                        bbox_inches="tight"
                        )
        plt.show()
    
    
    @staticmethod
    def print_n_model_evals(source: Path,
                            top_n: int,
                            data_loader: DataLoader,
                            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            cols_to_print: List[str],
                            wt_id: Optional[List[str]] = None,
                            y_limits: Optional[Tuple[Tuple[float,float],Tuple[float,float]]]= None
                            ) -> pd.DataFrame:
        """_summary_

        Args:
            source (Path): _description_
            top_n (int): the top n models with best mean error on val set
            data_loader (DataLoader): ...
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Make sure the loss fn is returning element-wise losses. E.g. nn.MSE(reduction="none")
            cols_to_print (List[str]): The values of the signals you want to print.
            wt_id (Optional[List[str]], optional): The wt(s) you want to inspect. Defaults to None.
            y_limits (Optional[Tuple[Tuple[float,float],Tuple[float,float]]], optional): Make a gap in the y-axis: e.g. ((0,1),(5,7)). Defaults to None.

        Returns:
            pd.DataFrame: The validation is expensive, therefore the loss table is being returned.
        """
        # src = Path(source)
        
        # if not os.path.exists(src):
        #     raise ValueError(f"source: {source} does not exist.")
        
        # files = sorted(src.rglob("*.pth"))
        # if top_n > len(files):
        #     print(f"Not sufficient models found. Changing top_n from {top_n} to {len(files)}")
        #     top_n = len(files)
        # #[print(f) for f in files]
        # ae_list = [tl.load_autoencoder(device, f) for f in files]
        
        # ae_list.sort(key=lambda e: e[2]["best_val"])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ae_list = tl.get_model_results(src=source)
        
        for i in tqdm(range(top_n)):
            ae, _, ckpt, _, _ = ae_list[i] # model
            fn = os.path.basename(ckpt["file_path"])
            df_eval = tl.eval_model(
                    model=ae,
                    data_loader=data_loader,
                    device= device,
                    loss_fn=loss_fn)
            
            for col in cols_to_print:
                PrettyPrint.print_loss(df_eval,
                                    save_dir=None,
                                    values=col,
                                    title=f"{col}: filename= {fn}",
                                    wt_id=wt_id,
                                    y_limits= y_limits)
            return df_eval
    
    @staticmethod
    def plot_signal_or_compare(
        df1: pd.DataFrame,
        sig: str,
        wt_id: int,
        df2: Optional[pd.DataFrame] = None,
        label1: str = "Original",
        label2: str = "Manipulated",
        anomaly_window: Optional[Tuple[str| pd.Timestamp, str | pd.Timestamp]] = None,
        y_limits: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        dpi: int = 240,
        file_path: Optional[Path] = None,
    ):
        """Plots one or two signals to compare

        """
        
        df1 = df1[df1[WT_ID].astype(int) == int(wt_id)]
        
        if df2 is not None:
            df2 = df2[df2[WT_ID].astype(int) == int(wt_id)]
        
        
        
        def prep(df: Optional[pd.DataFrame])-> Optional[pd.DataFrame]:
            if df is None:
                return None
            sig_df = df[[TS_COL, sig]].copy()
            if not np.issubdtype(sig_df[TS_COL].dtype, np.datetime64):
                sig_df[TS_COL] = pd.to_datetime(sig_df[TS_COL], errors="raise")
            
            sig_df.sort_values(TS_COL, inplace=True)
            return sig_df
            
        S1 = prep(df1)
        S2 = prep(df2)
        
        fig, ax = plt.subplots(figsize=(16,10))
        
        
        if S2 is not None and not S2.empty:
            ax.plot(S2[TS_COL].to_numpy(), S2[sig].to_numpy(), linewidth=1.0, label=label2)
        
        ax.plot(S1[TS_COL].to_numpy(), S1[sig].to_numpy(), linewidth=1.0, label=label1)
        
        ax.xaxis.set_major_locator(mpdates.MonthLocator())
        ax.xaxis.set_major_formatter(mpdates.DateFormatter("%Y-%m"))
        
        if y_limits is not None:
            ax.set_ylim(*y_limits)
            
        if anomaly_window is not None:
            start, end = anomaly_window
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.2)
            
        ax.set_xlabel("Time")
        ax.set_ylabel(sig)
        ax.set_title(title or sig)
        ax.grid(True, alpha=0.3)
        
        ax.legend(loc="upper right", frameon=True)
        
        plt.tight_layout()
        if (file_path is not None):
            if not str(file_path).endswith(".png"):
                raise ValueError("file need to be png")
            plt.savefig(file_path, 
                        dpi=dpi, 
                        bbox_inches="tight"
                        )
        plt.show()
        
        return ax
    
    @classmethod 
    def print_powercurve(cls,
        df_pc: pd.DataFrame,
        df_wts: pd.DataFrame,
        df_detections: Optional[pd.DataFrame] = None,
        detections_label: str = "Detection",
        wind_col: str = "Wind speed (m/s)",
        power_col: str = "Power (kW)",
        title: str = "Farm Power Curve - Test Period",
        dpi: int = 240,
        save_dir: Optional[Path]= None,
        file_name: Optional[str] = None,
        show: bool = True,
        highlight_wt: Optional[int] = None,
        highlight_ts_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        alpha_det: float = 0.15,
        alpha_ts_range: float = 0.9,
        fig_size: Tuple[int,int] = (9,4),
        xlabel= "Wind speed (m/s) at hub height"
    ) -> None:
        
        if df_pc.empty:
            raise ValueError("df_pc is empty.")
        if df_wts.empty:
            raise ValueError("df_wts is empty")
        
        pairs = [(df_wts, "df_wts")]
        if df_detections is not None:
            pairs.append((df_detections, "df_detections"))
            
        
        for df_, name in pairs:
            missing = [c for c in (wind_col, power_col) if c not in df_.columns]
            if missing:
                raise KeyError(f"{name}: missing columns {missing}")
        
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        x_pc = df_pc.index.to_numpy(dtype=float)
        y_pc = df_pc["power_norm"].to_numpy(dtype=float)
        ax.plot(
            x_pc,
            y_pc,
            label="Std. Power Curve",
            linewidth=0.8,
            color="black",
            zorder=3,
        )
        
        x_points = df_wts[wind_col].to_numpy(dtype=float)
        y_points = df_wts[power_col].to_numpy(dtype=float)
        
        ax.scatter(
            x_points,
            y_points,
            s=2, # marker,
            alpha=0.15,
            label="Measurements",
            zorder=1
        )
        
        if df_detections is not None and not df_detections.empty:
            x_det = df_detections[wind_col].to_numpy(dtype=float)
            y_det =df_detections[power_col].to_numpy(dtype=float)
            
            ax.scatter(
               x_det,
               y_det,
               s=30,
               alpha=alpha_det,
               marker="x",
               color="red",
               label=detections_label,
               zorder=2, 
            )
        
        if highlight_wt is not None:
            df_high = df_wts[df_wts[ic.WT_ID]  == int(highlight_wt)].copy()
            
            if highlight_ts_range is not None:
                h_start, h_end = highlight_ts_range
                h_start = pd.to_datetime(h_start)
                h_end = pd.to_datetime(h_end)
                
                df_high = df_high[
                    (pd.to_datetime(df_high[ic.TS_COL]) >= h_start) &
                    (pd.to_datetime(df_high[ic.TS_COL]) <= h_end)
                ]
            if not df_high.empty:
                t = pd.to_datetime(df_high[ic.TS_COL])
                t_num = t.view("int64").to_numpy()
                
                if highlight_ts_range is not None:
                    h_start, h_end = highlight_ts_range
                    h_start = pd.to_datetime(h_start)
                    h_end = pd.to_datetime(h_end)
                    
                    c_ts = h_start + (h_end - h_start) / 2
                    c_num =int(pd.to_datetime(c_ts).value)
                    
                    half_span_ns = int((h_end.value - h_start.value) // 2)
                    if half_span_ns <= 0:
                        t_norm = np.full_like(t_num, 0.5, dtype=float)
                    else:
                        rel = (t_num.astype(np.int64) - c_num) / float(half_span_ns)
                        rel = np.clip(rel, -1.0, 1.0)
                        t_norm = (rel + 1.0) / 2.0
                else:
                    if t_num.ptp() == 0:
                        t_norm = np.zeros_like(t_num, dtype=float)
                    else: 
                        t_norm = (t_num - t_num.min()) / (t_num.ptp())
                
                center_red_cmap = LinearSegmentedColormap.from_list(
                    "center_red",
                    [(0.0, "blue"), (0.5, "red"), (1.0, "yellow")],
                )
                
                sc = ax.scatter(
                    df_high[wind_col].to_numpy(dtype=float),
                    df_high[power_col].to_numpy(dtype=float),
                    c=t_norm,
                    cmap=center_red_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    marker="x",
                    s=30,
                    linewidth=1.0,
                    alpha=alpha_ts_range,
                    zorder=1,
                    label=f"WT {highlight_wt} within Window "
                )
                cbar = fig.colorbar(sc, ax=ax)
                
                if highlight_ts_range is not None:
                    cbar.set_ticks([0.0, 0.5, 1.0])
                    cbar.set_ticklabels(["start", "center", "end"])
                    cbar.set_label("Time within Window")
                else:
                    cbar.set_label("Time (normalized 0 = early, 1 = late)")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(power_col)
        ax.set_title(title)
        
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper right", frameon=True)
        
        #ax.set_xlim(float(x_pc.min(), float(x_pc.max())))
        
        fig.tight_layout()
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if file_name is None:
                file_name = "part2_pc_and_detections.png"
            fpath  =save_dir / file_name
            fig.savefig(fpath, dpi=dpi)
            
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    @classmethod
    def plot_wind_vs_nacelle(cls,
        df: pd.DataFrame,
        ts_col: str = ic.TS_COL,
        wind_col: str = "Wind direction (°)",
        nacelle_col: str = "Nacelle position (°)",
        ts_mark: Optional[pd.Timestamp| str] = None,
        ts_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        title: str = "Wind Direction vs. Nacelle Position",
        wind_speed_col: Optional[str] = None,
        wind_speed_label: str ="Wind speed (m/s)",
        power_col: str = "Power (kW)",
        power_label: str = "Power (kW)",
        save_filename: Optional[str] = None,
        dpi: int = 300,
        ax: Optional[plt.Axes] = None,
        show: bool = True
        ):
        
        if df.empty:
            return
        
        df = df.copy()
        
        df[ts_col] = pd.to_datetime(df[ts_col])
        
        if ts_range is not None:
            start, end = ts_range
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            
            mask = (df[ts_col]>= start) & (df[ts_col] <= end)
            df = df.loc[mask]
            
            if df.empty:
                return
        
        created_fig = False
        fig = None
        if ax is None:
             
            fig, ax = plt.subplots(figsize=(10,4))
            created_fig = True
        
        delta  =(df[wind_col].astype(float) - df[nacelle_col].astype(float)).abs()
        l1= ax.plot(
            df[ts_col], 
            delta, 
            label=f"|Wind_direction - Nacelle pos.| ", 
            linewidth=0.7)
        
        ax.set_xlabel("Time") 
        ax.set_ylabel("Absoulute Angle difference (°)")
        ax.set_title(title)
        ax.set_ylim(0, 360)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        
        lines = l1
        labels=[f"|Wind dir. - nacelle pos.|"]
        ax2 = None
        if power_col is not None and power_col in df.columns:
            ax2 =ax.twinx()
            l2 = ax2.plot(
                df[ts_col],
                df[power_col].astype(float),
                linewidth=0.7,
                alpha=0.7,
                label=power_label,
                color="tab:orange"
            )
            ax2.set_ylabel(power_label)
            lines += l2
            labels += [power_label]
            
        if wind_speed_col is not None and wind_speed_col in df.columns:
            ax2 = ax.twinx()
            l3= ax2.plot(
                df[ts_col],
                df[wind_speed_col],
                linewidth=0.7,
                alpha=0.7,
                label=wind_speed_col,
                color="tab:green"
            )
            ax2.set_ylabel(wind_speed_label)
            
            lines  += l3
            labels += [wind_speed_label]
        
        ax.legend(lines, labels, loc="upper left")
        if ts_mark is not None:
            ts_mark = pd.to_datetime(ts_mark)
            ax.axvline(ts_mark, linestyle="--", linewidth=1.0, color="red")
        
        if created_fig: 
            fig.tight_layout()
        if (save_filename is not None):
            fig.savefig(ic.PATH_PRINTS / save_filename, 
                        dpi = dpi, 
                        bbox_inches="tight"
                        )
        if created_fig:
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        return ax
    
    @classmethod
    def load_farm_impute_spans(cls,
                               global_start: pd.Timestamp,
                               global_end: pd.Timestamp,
                               time_step: pd.Timedelta = pd.Timedelta(minutes=10),
                               wt_ids: Optional[List[int]] = None,
                                ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        global_start = pd.to_datetime(global_start)
        global_end = pd.to_datetime(global_end)
        
        if global_end < global_start:
            raise ValueError("global_start is after global_end")
        
        idx_full = pd.date_range(start=global_start, end=global_end, freq=time_step)
        
        if wt_ids is None:
            wt_ids = [1,2,4,5,6,7,8,9,10,11,12,13,14,15]
        
        files = glob(os.path.join(ic.PATH_IMPUT_MASKS, "*.parquet"))
        spans_per_wt: dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        
        signal_key = ANY
        canditate = [signal_key]
        
        for wt in wt_ids:
            file_idx = cls.find_wt_file(files, wt)
            if file_idx is None:
                raise ValueError(f"file with wt_id {wt} not found.")
            
            df_impute_log = pd.read_parquet(files[file_idx])
            df_impute_log[ic.TS_COL] = pd.to_datetime(df_impute_log[ic.TS_COL], errors="raise")
            
            df_impute_log = df_impute_log[
                (df_impute_log[ic.WT_ID] == int(wt)) &
                (df_impute_log[ic.TS_COL] >= global_start) &
                (df_impute_log[ic.TS_COL] <= global_end) &
                (df_impute_log["Signal"].isin(canditate))
            ]
            
            if df_impute_log.empty:
                spans_per_wt[wt] = []
                continue
            
            ts_events = pd.to_datetime(df_impute_log[ic.TS_COL]).values
            mask = np.isin(idx_full, ts_events)
            
            spans_per_wt[wt] = cls._bool_runs_to_spans_fixed_step(
                idx_full,
                mask,
                step=time_step,
                inclusive_end=True,
            )
            
            all_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for wt in wt_ids:
                all_spans.extend(spans_per_wt.get(wt, []))
            
            all_spans = cls._clip_spans_to_range(all_spans, global_start, global_end)
            merged_spans = cls._merge_spans(all_spans)
            
            return merged_spans
            
    @classmethod
    def plot_dataset_time_periods(cls,
        time_periods: dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
        figsize: Tuple[int, int] = (10, 2.5),
        title: str = "Dataset splits over time",
        global_start: Optional[pd.Timestamp] = None,
        global_end: Optional[pd.Timestamp] = None,
        save_path: str | None = None,
        dpi: int = 300,
        time_step: pd.Timedelta = pd.Timedelta(minutes=10),
        show_impute: bool = True,
        impute_label: str = "Data Imputation"
        ):
        global_start = pd.to_datetime(global_start)
        global_end = pd.to_datetime(global_end)
        
        if global_end < global_start:
            raise ValueError("global start is after global end.")
        
        default_colors = {
            "Train": "#1f77b4",
            "Val1": "#d38a4a",
            "Val2": "#6ebd07",
            "Test": "#7327d6"
        }
        
        total_samples = (int((global_end - global_start) // time_step)) +1
        
        if total_samples <= 0:
            raise ValueError("Total samples must be greater than 0")
        
        
        n_samples: dict[str, int] = {}
        for name, (start,end) in time_periods.items():
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if end < start:
                raise ValueError(f"start after end.")
            n = int((end -start) // time_step) +1 #inclusive
            n_samples[name] = n
        
        
        
        split_percents: dict[str, float] = {
            name: 100.0 * n / total_samples for name, n in n_samples.items()
        }
        
        fig, ax  = plt.subplots(figsize=figsize)
        
        y_min, y_max = 0,1
        
        for name, (start, end) in time_periods.items():
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            
            share = split_percents[name]
            
            color = default_colors.get(name, None)
            
            ax.axvspan(
                start,
                end,
                ymin=0.10,
                ymax=0.90,
                color=color,
                label=name,
            )
            
            mid = start + (end-start)/ 2
            ax.text(
                mid,
                0.5,
                f"({share:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
            )
        
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("Time")
        
        boundaries: list[pd.Timestamp] = []
        for (start, end) in time_periods.values():
            boundaries.append(pd.to_datetime(start))
            boundaries.append(pd.to_datetime(end))
        
        if global_start is None:
            global_start = min(boundaries)
            global_end = max(boundaries)
        
        boundaries.append(pd.to_datetime(global_start))
        boundaries.append(pd.to_datetime(global_end))
        
        boundaries = sorted(pd.unique(boundaries))
        
        ax.set_xticks(boundaries)
        date_fmt = mpdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate(rotation=45, ha="right")
        
        for ts in boundaries:
            ax.axvline(ts, ymin=0, ymax=1, color="black", linewidth=0.5, alpha=1)

        if show_impute:
            impute_spans = PrettyPrint.load_farm_impute_spans(
                global_start=global_start,
                global_end = global_end,
                time_step=time_step,
                wt_ids=None,
            )
            
            spans_clipped = cls._clip_spans_to_range(
                impute_spans, global_start, global_end
            )
            
            
            impute_samples= 0
            for start, end in spans_clipped:
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                if end < start:
                    continue
                
                impute_samples += int((end-start) // time_step) + 1 
            
            impute_share = 100.0 * impute_samples / total_samples
            impute_label_with_share = f"{impute_label} ({impute_share:.1f}%)"
            
            first = True
            for start, end in spans_clipped:
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                if end < start:
                    continue
                
                ax.axvspan(
                    start,
                    end,
                    ymin=0.05,
                    ymax=0.95,
                    alpha=0.2,
                    color="red",
                    label=impute_label_with_share if first else "_no_legend_",
                )
                
                if first:
                    first = False
        
        
        handles, _ = ax.get_legend_handles_labels()
        
        if handles:
            ax.legend(
                loc="upper left",
                #bbox_to_anchor=(0.5, 1.15),
                ncol=len(handles),
                frameon=True,
                framealpha=1.0
            )
        
        # ax.legend(
        #         loc="upper left",
        #         # bbox_to_anchor=(0.5, 1.15),
        #         ncol=1,
        #         frameon=True,
        #     )
        ax.set_xlim(global_start, global_end)
        ax.margins(x=0)
        
        plt.tight_layout()
        
        if save_path is not None:
            
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            
        plt.show()
        
    
    @classmethod
    def unit(cls, s: str):
        m = re.search(r"\(([^)]*)\)s*$", s)
        return m.group(1) if m else None
    
    @classmethod
    def plot_signals_compact(cls,
        df: pd.DataFrame,
        wt_id: int,
        signals: List[str],
        ts_col: str = ic.TS_COL,
        wt_col: str = ic.WT_ID,
        ts_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        ts_mark: Optional[pd.Timestamp|str] = None,
        ts_mark_label= "Detection",
        anomaly_spans: Optional[list[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        fig_size: Tuple[int,int] = (2.5,1.25),
        save_path: Path = Path(ic.PATH_PRINTS / "P2_signals_compact.png"),
        dpi: int = 300,
        add_pre_post_mean: bool = True,
        mean_linewidth: float = 1.2,
        impute_use_any: bool = True,
        impute_alpha: float = 0.2,
        impute_zorder: float = 0.5,
        impute_color: str = "red",
        plot_power: bool = False,
        power_col: str = "Power (kW)",
        power_col_ylabel: str = "kW",
        power_alpha: float = 0.1,
        power_linewidth: float = 0.5,
        sig_linewidth: float = 0.2,
        
    ):
        if df.empty:
            raise ValueError("df is empty")
        
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col],errors="raise")
        
        df = df[df[wt_col].astype(int) == int(wt_id)].sort_values(ts_col)
        
        if ts_range is not None:
            start, end = pd.to_datetime(ts_range[0]),pd.to_datetime(ts_range[1])
            df = df[(df[ts_col] >= start) & (df[ts_col] <= end)]
            if df.empty:
                raise ValueError(f"df is empty")
        signals = [s for s in signals if s in df.columns]
        if not signals:
            raise ValueError("No signals match with df cols")
        
        n = len(signals)
        fig_w, fig_h = fig_size
        fig_h_used = fig_h*n
        figsize_used  = (fig_w, fig_h_used)
        # if fig_size is not None:
        #     figsize_used = (fig_size[0], fig_size[1])
        # else:
        #     figsize_used = (12, max(3.0, 0.6 * n))
            
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            sharex=True,
            figsize=figsize_used,
            gridspec_kw={"hspace": 0.15},
        )
        
        axes = np.atleast_1d(axes)
        
        if ts_mark is not None:
            if isinstance(ts_mark, (list,tuple, np.ndarray, pd.Series)):
                if len(ts_mark) == 0:
                    ts_mark = None
                else:
                    ts_mark = ts_mark[0]
                ts_mark = pd.to_datetime(ts_mark)
                
            ts_mark = pd.to_datetime(ts_mark)
        
        spans_by_signal: dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        
        files = glob(os.path.join(PATH_IMPUT_MASKS, "*.parquet"))
        file_idx = cls.find_wt_file(files, wt_id)
        if file_idx is None:
            raise ValueError(f"file with wt_id {wt_id} not found")
        
        df_impute_log = pd.read_parquet(files[file_idx])
        df_impute_log[ts_col] = pd.to_datetime(df_impute_log[ts_col], errors="coerce")
        
        if ts_range is not None:
            x_min = pd.to_datetime(ts_range[0])
            x_max = pd.to_datetime(ts_range[1])
        else:
            x_min = pd.to_datetime(df[ts_col].min())
            x_max = pd.to_datetime(df[ts_col].max())
            
        idx_plot = pd.to_datetime(df[ts_col]).sort_values()
        
        df_impute_log = df_impute_log[
            (df_impute_log[WT_ID].astype(int) == int(wt_id)) &
            (df_impute_log[ts_col] >= x_min) & (df_impute_log[ts_col] <= x_max)
        ].copy()
        
        for sig in signals:
            l = []
            if impute_use_any:
                l.append(ANY)
            l.append(str(sig))
            if str(sig).startswith(f"{RE_PREFIX}"):
                l.append(str(sig)[len(RE_PREFIX):])
            
            df_sig = df_impute_log[df_impute_log["Signal"].isin(l)]
            if df_sig.empty:
                spans_by_signal[str(sig)] = []
                continue
            
            ts_events = pd.to_datetime(df_sig[ts_col]).values
            mask = np.isin(idx_plot.values, ts_events)

            spans = cls._bool_runs_to_spans_fixed_step(
                idx_plot,
                mask,
                step=pd.Timedelta(minutes=10),
                inclusive_end=True,
            )
            
            spans = cls._clip_spans_to_range(spans, x_min, x_max)
            spans = cls._merge_spans(spans)
            
            spans_by_signal[str(sig)] = spans
            
        
        for i, sig in enumerate(signals):
            ax = axes[i]
            y = pd.to_numeric(df[sig], errors="coerce")
            t= pd.to_datetime(df[ts_col])
            
            ax.plot(t,y, linewidth=sig_linewidth, label=f"WT {wt_id}")
            ax2 = None
            if plot_power and (power_col in df.columns):
                ax2 = ax.twinx()
                y_pow = pd.to_numeric(df[power_col], errors="coerce")
                ax2.plot(
                    t, y_pow,
                    linewidth=power_linewidth,
                    alpha=power_alpha,
                    label=power_col,
                    zorder=0.1,
                    color="green"
                )
                ax2.grid(False)
                ax2.set_ylabel(power_col_ylabel, fontsize=8)
                
            spans = spans_by_signal.get(str(sig), [])
            if spans:
                cls._add_spans(
                    ax,
                    spans,
                    label="Data Imputation",
                    alpha=impute_alpha,
                    zorder = impute_zorder,
                    add_legend = True,
                    color=impute_color,
                )
            
            if not add_pre_post_mean and (ts_mark is None):
                raise ValueError(f"if add_pre_post_mean is true then ts_mark needs to be set")
            tm = pd.to_datetime(ts_mark)
            t_left = t.min()
            t_right = t.max()
            m_win = (t >= t_left) & (t <= t_right)
            t_win = t[m_win]
            y_win = y[m_win]
            
            m_pre = (t_win < tm)
            m_post = (t_win >= tm)
            
            pre_vals = y_win[m_pre].to_numpy(dtype=float)
            post_vals = y_win[m_post].to_numpy(dtype=float)
            
            pre_vals = pre_vals[np.isfinite(pre_vals)]
            post_vals = post_vals[np.isfinite(post_vals)]
            
            if pre_vals.size > 0:
                median_pre = float(np.median(pre_vals))
                ax.axhline(median_pre, 
                           linestyle="--", 
                           linewidth=mean_linewidth, 
                           alpha=0.9, 
                           label=f"Median pre={median_pre:.2f}",
                           color="purple",
                           zorder=5,
                           )
            
            if post_vals.size > 0:
                median_post = float(np.median(post_vals))
                ax.axhline(median_post, 
                           linestyle=":", 
                           linewidth=mean_linewidth, 
                           alpha=0.9, 
                           label=f"Median post={median_post:.2f}",
                           color="orange",
                           zorder=5,
                           )
            
            
            unit = cls.unit(sig)
            ax.set_ylabel(unit if unit else "", fontsize=9)
            ax.set_title(sig, loc="left", fontsize=9, pad=2)
            ax.grid(True, linewidth=0.5, alpha=0.4)
            
            if anomaly_spans:
                for (s,e) in anomaly_spans:
                    ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.12)
            
            if ts_mark is not None:
                ax.axvline(ts_mark, linestyle="--", linewidth=2.0, color="red", label=ts_mark_label)
            h1, l1 = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                h1 += h2
                l1 += l2
            leg = ax.legend(h1, l1, loc="upper center", fontsize=8, frameon=True, ncol=len(l1), handlelength=3.0)
            
            for line in leg.get_lines():
                line.set_linewidth(3.0)
        
            if ax2 is None:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + 10)
            else:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + 20)
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin, ymax + 500)
            
        axes[-1].set_xlabel("Time")
        
       
        if save_path is not None:
            
            fig.savefig(save_path, dpi=dpi)
        plt.show()
        
        
    @classmethod
    def plot_heatmap_RE(cls,
                     df: pd.DataFrame,
                     windows: list,
                     window_alpha=0.1,
                     title:str="RE Heatmap",
                     wt_id: int=1,
                     filename="heatmap.png",
                     cmap: str= "magma",
                     figsize=(18,10),
                     dpi: int=300,
                     point_signal: str | None = None,
                     point_width: int = 2,
                     use_lognorm = False,
                     use_manual_color_scale = False,
                     vmin=0.0,
                     vmax=4.0,
    ):
        df = df.copy()
        df[ic.TS_COL] = pd.to_datetime(df[ic.TS_COL])
        df = df[df[ic.WT_ID] == wt_id].sort_values(ic.TS_COL).reset_index(drop=True)
        
        anom_start = min(pd.to_datetime(s) for s, _ in windows)
        anom_end = max(pd.to_datetime(e) for _,e in windows)
        df = df[ (df[ic.TS_COL] >= anom_start) & (df[ic.TS_COL] <= anom_end)]
        
        re_cols = sorted([col for col in df.columns if col.startswith(ic.RE_PREFIX)],reverse=True)+[ic.MEAN_LOSS_PER_SAMPLE]
        heatmap = df[re_cols].to_numpy(dtype=float).T
        y_labels = [col.replace(ic.RE_PREFIX, "", 1) for col in re_cols]
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
       
        
        ax.set_yticks(np.arange(len(y_labels)))
        
        
        y_labels = ["MSE" if t == ic.MEAN_LOSS_PER_SAMPLE else t for t in y_labels]
        ax.set_yticklabels(y_labels, fontsize=8) 
        
        num_times = len(df)
        xtick_idx = np.linspace(0, num_times - 1, num=min(10, num_times), dtype=int)
        xtick_labels = df.iloc[xtick_idx][ic.TS_COL].dt.strftime("%Y-%m-%d\n%H:%M")
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ts = df[ic.TS_COL].to_numpy()
        
        if point_signal is not None:
            p_col = f"{ic.RE_PREFIX}{point_signal}"
            point_row_idx = re_cols.index(p_col)
            mse_row_idx = re_cols.index(ic.MEAN_LOSS_PER_SAMPLE)
            for s,e in windows:
                s = pd.to_datetime(s)
                e = pd.to_datetime(e)
                if s == e:
                    p_idx  = np.searchsorted(ts, np.datetime64(s), side="left")
                    if 0 <= p_idx < heatmap.shape[1]:
                        left = max(0,p_idx - point_width)
                        right = min(heatmap.shape[1] - 1, p_idx + point_width)
                        heatmap[point_row_idx, left:right+1] = heatmap[point_row_idx, p_idx]
                        heatmap[mse_row_idx, left:right+1] = heatmap[mse_row_idx, p_idx]
       
        
        for s,e in windows:
            s_idx = np.searchsorted(ts, np.datetime64(s), side="left")
            e_idx = np.searchsorted(ts, np.datetime64(e), side="right")-1
            # ax.axvline(s_idx - 5, color="cyan", linestyle="--", linewidth=1.0)
            # ax.axvline(e_idx + 5, color="cyan", linestyle="--", linewidth=1.0)
            ax.axvspan(s_idx - 2, e_idx + 2, color="cyan", alpha=window_alpha)
        
        if use_lognorm:
            im = ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                norm=LogNorm(vmin=1e-4, vmax=np.max(heatmap)),
            )
        elif not use_lognorm and not use_manual_color_scale: 
            im = ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=np.quantile(heatmap, 0.99)
            )
        elif not use_lognorm and use_manual_color_scale:
            im = ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Signal")
        
        ax.set_title(title)
        cbar= fig.colorbar(im, ax=ax)
        cbar.set_label("RE")
        
        fig.tight_layout()
        if filename is not None:
            fig.savefig(Path(ic.PATH_PRINTS) / filename, bbox_inches="tight")
        
        plt.show()
        
    
    @classmethod
    def plot_corr_matrix(cls,
                         sigs:list[str],
                         df:pd.DataFrame,
                         title="Empirical Pearson Correlation of the Training Set Time Period",
                         figsize=(9,9),
                         cbar_kws={"shrink":0.35, "pad":0.05, "aspect":30},
                         filename="part1_corr_matrix.png",
                         x_label_rotation = 20,
    ):
        cols = [col for col in sigs if col in df.columns]
        num_subset = df[cols]
        corr_mat = num_subset.corr()

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            corr_mat,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            annot_kws={"size": 12},
            cbar_kws=cbar_kws
            
        )
        plt.title(title, fontsize=18)
        
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        
        plt.setp(ax.get_xticklabels(), rotation=x_label_rotation, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(Path(ic.PATH_PRINTS / filename), dpi=300, bbox_inches="tight")
        plt.show()
    
    