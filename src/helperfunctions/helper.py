
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from glob import glob
from helperfunctions.intern_constants import PATH_IMPUTED, TRAIN_END, PATH_PC_FILTERING, TRAIN_START, PATH_MINMAX_SCALER, PATH_FEATURE_ORDER_FILE, TS_COL, WT_ID
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict, Any, cast
import joblib
import json
import numpy as np
import os
import pandas as pd
import torch


TensorType = torch.Tensor
Tensor_fn_type = Callable[[TensorType], TensorType]

Transform_sig = Union[Tensor_fn_type, Sequence[Tensor_fn_type]]
DFTransformation_fn_type = Callable[[pd.DataFrame], pd.DataFrame]

class MultiCSVDataset(Dataset):
    """ Custom DataSet. Returns: 
        X:torch.Tensor, uid: int - Unique IDentifier is used to map x to a timestamp and wt_id 

    """
    def __init__(
        self,
        dataset_paths: Sequence[Path], # list of all csvs of a specific split
        signal_cols: Sequence[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None, 
        ts_col: str = TS_COL,
        *,
        transformations: Union[Transform_sig, Sequence[Transform_sig], None ] = None,
        filter_fns: Union[DFTransformation_fn_type, Sequence[DFTransformation_fn_type], None] = None,
        dtype: np.dtype = np.float32, # target data type
        chunk_size: Optional[int] = None
    ):
        
        if ((not dataset_paths) or (not signal_cols)):
            raise ValueError(f"Initialisation of MultiCSVDataset Class failed. Check Parameters")

        #TODO
        #load cols from train data -> same order as train data - in respect to the minmax scaler
        #selected_cols = list(dict.fromkeys(signal_cols  + [WT_ID,ts_col]))
        selected_cols = [ts_col, WT_ID, *list(signal_cols)]
        
        dfs: list[pd.DataFrame] = []
        
        for file_path in dataset_paths:
            if chunk_size is None:
                df = pd.read_csv(
                    file_path,
                    parse_dates=[ts_col],
                )
                df = self._filter_ts(df, start_time, end_time, ts_col=ts_col)
                
                missing = [col for col in selected_cols if col not in df.columns]
                if missing:
                    raise ValueError(
                        f"{os.path.basename(file_path)}: missing columns {missing}"
                        f"Available: {list(df.columns)}"
                    )
                # print(f"df columns = {df.columns}"
                #       f"selected_cols = {selected_cols}")
                df = df.reindex(columns=selected_cols)
                
                dfs.append(df)
            else:
                for chunk in pd.read_csv(
                    file_path,
                    parse_dates=[ts_col],
                    usecols=selected_cols,
                    chunksize=chunk_size
                ):
                    chunk = self._filter_ts(chunk, start_time, end_time, ts_col=ts_col)
                    
                    missing = [col for col in selected_cols if col not in chunk.columns]
                    if missing:
                        raise ValueError(
                            f"{os.path.basename(file_path): missing columns {missing}}"
                            f"Available: {list(chunk.columns)}"
                        )
                    
                    chunk = chunk.reindex(columns=selected_cols)
                    if not chunk.empty:
                        dfs.append(chunk)
        if not dfs:
            raise ValueError(f"No data found given time period: start_time={start_time}, end_time={end_time}")        
        
        # axis=0 : create global index - will be used later to identify ts and wt_id
        df_complete = pd.concat(dfs, ignore_index=True, axis=0)
        
        ts_ns = (df_complete[ts_col]
                 .astype("datetime64[ns]")
                 .astype(np.int64, copy=False))
        wt:np.int64 = df_complete[WT_ID].astype(np.int64)
        # unlikely to have collisions here - TODO: may be to be explained
        df_complete["Global index"] = ts_ns + wt
        
        if filter_fns is None:
            filter_fn_list: List[DFTransformation_fn_type] = []
        elif isinstance(filter_fns, Sequence):
            filter_fn_list = list(filter_fns)
        else:
            filter_fn_list = [filter_fns]
            
        for fn in filter_fn_list:
            df_complete = fn(df_complete)
        
        if "Global index" not in df_complete.columns:
            raise RuntimeError("Global index col was dropped by a filter")
        
        if ts_col not in df_complete.columns:
            raise KeyError(f"TS col {ts_col!r} is missing after filtering")
        
        
        if WT_ID not in df_complete.columns:
            raise KeyError(f"Column 'WT_ID' is missing after filtering")
        
        #error handling
        miss = [col for col in signal_cols if col not in df_complete.columns]
        if miss:
            raise ValueError(f"Missing signal_cols: {miss}")
        
        
        
        #set final global range index
        df_complete = df_complete.reset_index(drop=True)
        
        self.features = df_complete[list(signal_cols)].to_numpy(dtype=dtype)
        self._ts_ns = df_complete[ts_col].astype("datetime64[ns]").astype("int64", copy=False)
        self._wt = df_complete[WT_ID].astype(np.int16).to_numpy()
        self._uids = df_complete["Global index"].to_numpy(copy=False)
        self._uid_to_pos: Dict[int, int] = {int(u): i for i,u in enumerate(self._uids)}
        
        if transformations is None:
            self.transformations: List[Transform_sig] = []
        elif isinstance(transformations, Sequence):
            self.transformations = list(transformations)
        else:
            self.transformations = [transformations]
        
        self._signal_cols = list(signal_cols)
        
    def to_df(self, ts_col: str = TS_COL) -> pd.DataFrame:
        ts = pd.to_datetime(self._ts_ns, unit="ns")
        df = pd.DataFrame(self.features, columns=list(self._signal_cols))
        df.insert(0, WT_ID, self._wt.astype(np.int16))
        df.insert(0, ts_col, ts)
        
        return df
    
    
    def update_from_df(
        self,
        df_new: pd.DataFrame,
    ) -> None:
        """ Updates Dataset in-place\n
        
        Be aware: 
        - Do not change original column ordering.
        - Do not change the total row number, Ordering, Timestamps or WT_IDs
        - df_new must be originated from self.to_df!!
        - Only changes in values of signals are allowed
        
        Otherwise the update may be faulty. 
        """
        
        #small checks
        if len(df_new) != len(self):
            raise ValueError("Difference in row numbers in new_df. Check docstring of update_from_df().")
    
        upd_cols = [c for c in df_new.columns if c in self._signal_cols]
        if not upd_cols:
            raise ValueError("No Columns to update. Check your df_new")
        
        # position based
        col_pos = {col: index for index,col in enumerate(self._signal_cols)}
        for col in upd_cols:
            # self.features : [N,F]
            self.features[:, col_pos[col]] = df_new[col].to_numpy(dtype=self.features.dtype, copy=False)
        

    
    @staticmethod
    def _filter_ts(df: pd.DataFrame, start_time: Optional[str], end_time: Optional[str], ts_col: str = TS_COL,):
        
        if ts_col not in df.columns:
            raise KeyError(f"Timestamp col: {ts_col} is missing in dataframe.")
        
        if  not np.issubdtype(df[ts_col].dtype, np.datetime64):
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
            
        if start_time is not None:
            df = df[df[ts_col] >= pd.to_datetime(start_time)]
        
        if end_time is not None:
            df = df[df[ts_col] <= pd.to_datetime(end_time)]
        
        return df
            
    
    def __len__(self) -> int:
        return self.features.shape[0]
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.features[index])
        for t in self.transformations:
            x = t(x) 
        uid = int(self._uids[index])
        return x,uid
    
    
def _get_dataset_paths(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.csv"))

def make_time_groups_from_dataset(
    dataset: MultiCSVDataset,
    # ts_key: str = TS_COL,
    # wt_key: str = WT_ID,
    # use_ns: bool = True
)-> Dict[int, List[int]]:
    
    groups: Dict[int, List[int]] = defaultdict(list)
    ts_ns = dataset._ts_ns
    
    for i,tsn in enumerate(ts_ns):
        groups[int(tsn)].append(i)
    
    # sort to wt_id within every group
    for _, idxs in groups.items():
        idxs.sort(key=lambda j: int(dataset._wt[j]))
    
    return dict(groups)

# Sampler
class  GroupedBatchSampler(Sampler[List[int]]):
    """ Returns per Iteration, indices for k timestamps (all turbines).
        A Shuffle mode is also provided.
    """
    
    def __init__(
        self,
        groups_list: List[List[int]],
        timestamps_per_batch: int = 18,
        shuffle: bool = True,
        seed: Optional[int] = 32,
    ):
        assert timestamps_per_batch >= 1# minimal number
        self.groups_list = groups_list
        self.timestamps_per_batch = timestamps_per_batch
        self.shuffle = shuffle
        self.numb_Gen = torch.Generator()
        self.numb_Gen.manual_seed(seed)
    
    def __iter__(self):
        n = len(self.groups_list)
        order = list(range(n))
        
        if self.shuffle:
            permutation = torch.randperm(n, generator=self.numb_Gen).tolist()
            order = [order[i] for i in permutation]
        
        for s in range(0,n,self.timestamps_per_batch):
            block = order[s: s+ self.timestamps_per_batch]
            flat = [idx for g in block for idx in self.groups_list[g]]
            if flat:
                yield flat
    
    def __len__(self):
        n = len(self.groups_list)
        # number of batches: round(n/K) = ceil((n+K-1)/K)
        return (n + self.timestamps_per_batch - 1) // self.timestamps_per_batch

class collate_timegroups:
    """ Collate for batches from (x, uid) - grouped by timestamp and sorted by wt.
        
        Returns:
        X, uids2d, timestamps_sorted, wt_order
    """
    def __init__(self, dataset:MultiCSVDataset):
        self.dataset = dataset
    
    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, List[List[int]], List[str], List[int]]:
        
        xs, uids = zip(*batch)
        U = np.asarray(uids, dtype=np.int64)
        
        # uid -> pos
        pos = np.array([self.dataset._uid_to_pos[int(u)] for u in U], dtype=np.int64)
        
        ts_ns = self.dataset._ts_ns[pos]
        wt = self.dataset._wt[pos].astype(int)
        
        by_ts: Dict[int, List[Tuple[torch.Tensor, int, int]]] = defaultdict(list)
        wt_set = set()
        
        for x, uid, tsn, w in zip(xs, U, ts_ns, wt):
            by_ts[int(tsn)].append((x, int(uid), int(w)))
            wt_set.add(int(w))
        
        timestamps_sorted_ns = sorted(by_ts.keys())
        wt_order = sorted(wt_set)
        
        X_list: List[torch.Tensor] = []
        uids2d: List[List[int]] = []
        n_wt_ref = None
        
        for tsn in timestamps_sorted_ns:
            items = by_ts[tsn]
            items.sort(key=lambda t: t[2]) #sort by wt
            x_tuple, uid_tuple, _ = zip(*items)
            
            if n_wt_ref is None:
                n_wt_ref = len(items)
            elif len(items) != n_wt_ref:
                raise ValueError("inconsistent numbers of wts between timestamp groups")
            
            X_list.append(torch.stack(list(x_tuple), dim=0)) # [WT, F]
            uids2d.append(list(uid_tuple))
            
        X = torch.stack(X_list, dim=0) # [TS, WT, F]
        timestamps_sorted = [pd.to_datetime(tsn).isoformat() for tsn in timestamps_sorted_ns]
        
        return X, uids2d, timestamps_sorted, wt_order

# wrapper
class collate_timegroups_with_transform:
    
    def __init__(self,
    dataset: MultiCSVDataset,
    batch_transform: Callable[[torch.Tensor, List[List[int]], List[str], List[int]], Tuple[torch.Tensor, any]]
    ):
        self.base = collate_timegroups(dataset) # returns X, uid2d, ts_sorted, wt_order
        self.batch_transform = batch_transform
        
    def __call__(self, batch):
        X, uids2d, ts_sorted, wt_order = self.base(batch)
    
        return self.batch_transform(X, uids2d)

class _FmMinmaxTransform:
    def __init__(self, dataset: MultiCSVDataset, apply_transf: bool = False):
        self.dataset = dataset
        self.apply_transf = apply_transf
        
        self.feature_cols = load_feature_order()
    
    def __call__(
        self,
        X: torch.Tensor,
        uids2d: List[List[int]],
    ):

        
        TS, WT, F = X.shape
        uids_flat = [u for group in uids2d for u in group]
        
        if self.apply_transf:
            # apply fm
            X = DataTransformations.substract_fleet_median_3d(X) # [TS,WT,F]
            X2d = X.reshape(-1, F) # [TS x WT, F]
            minmax_fn = DataTransformations.batch_minmax(feature_cols=self.feature_cols, dataset=self.dataset)
            # apply min max scaling
            X2d = minmax_fn(X2d, uids_flat)
            return X2d, uids_flat
        else:
            X2d = X.reshape(-1, F)
            return X2d, uids_flat

def make_fm_minmax_transform(dataset: MultiCSVDataset, apply_transf: bool = False):
    return _FmMinmaxTransform(dataset=dataset, apply_transf=apply_transf)
    
def _worker_init(wid: int):
    base = torch.initial_seed() % 2**32
    np.random.seed(base)
    torch.manual_seed(base)

def build_dataloaders(
    train_csv_dir: Path,
    val_csv_dir: Path,
    test_csv_dir: Path,
    cfg: TrainConfig,
    filter_fns: Union[DFTransformation_fn_type, Sequence[DFTransformation_fn_type], None] = None, 
    chunk_size: Optional[int] = None,
    num_workers: int = 8,
    pin_memory: bool = False,
    transformations: Union[Transform_sig, Sequence[Transform_sig], None] = None,
    *,
    min_group_size: int = 14, # penmanshiel 14 turbines
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train-, Val-, and TestLoader in one step.
    
    Returns:
        Tuple[DataLoader,DataLoader,DataLoader]: Train-, Val-, and TestLoader
    """
    signal_cols = cfg.signal_cols
    ts_col = cfg.ts_col
    
    train_ds = MultiCSVDataset(
        dataset_paths=_get_dataset_paths(train_csv_dir),
        signal_cols=signal_cols,
        ts_col=ts_col,
        transformations=transformations,
        filter_fns=filter_fns,
        #meta_cols=meta_cols,
        chunk_size=chunk_size
    )
    
    val_ds = MultiCSVDataset(
        dataset_paths=_get_dataset_paths(val_csv_dir),
        signal_cols=signal_cols,
        ts_col=ts_col,
        start_time=cfg.val_start_time,
        end_time=cfg.val_end_time,
        transformations=transformations,
        filter_fns=filter_fns,
        #meta_cols=meta_cols,
        chunk_size=chunk_size
    )
    
    test_ds = MultiCSVDataset(
        dataset_paths=_get_dataset_paths(test_csv_dir),
        signal_cols=signal_cols,
        ts_col=ts_col,
        start_time=cfg.test_start_time,
        end_time=cfg.test_end_time,
        transformations=transformations,
        filter_fns=filter_fns,
        #meta_cols=meta_cols,
        chunk_size=chunk_size
    )

    rng = None
    if cfg.seed is not None:
        rng = torch.Generator()
        rng.manual_seed(cfg.seed)
    
    #if group_by_time:
        # build time groups: Dict: ts_key -> List[int], within groups sort by WT_ID
    train_groups_dict = make_time_groups_from_dataset(train_ds)
    val_groups_dict = make_time_groups_from_dataset(val_ds)
    test_groups_dict = make_time_groups_from_dataset(test_ds)

    def dict_to_groups_list(dict: Dict[int, List[int]]) -> List[List[int]]:
        keys_sorted = sorted(dict.keys())
        groups_list = [dict[key] for key in keys_sorted if len(dict[key]) >= min_group_size]
        
        return groups_list
    
    train_groups_list = dict_to_groups_list(train_groups_dict)
    val_groups_list = dict_to_groups_list(val_groups_dict)
    test_groups_list = dict_to_groups_list(test_groups_dict)
    
    # create BatchSampler
    train_batch_sampler = GroupedBatchSampler(
        train_groups_list, timestamps_per_batch=cfg.batch_size, shuffle=True, seed= cfg.seed
    )
    
    val_batch_sampler = GroupedBatchSampler(
        val_groups_list, timestamps_per_batch=cfg.batch_size, shuffle=False, seed= cfg.seed
    )
    
    test_batch_sampler = GroupedBatchSampler(
        test_groups_list, timestamps_per_batch=cfg.batch_size, shuffle=False, seed= cfg.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_timegroups_with_transform(
            train_ds,
            make_fm_minmax_transform(train_ds, apply_transf=True)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=rng,
        worker_init_fn=_worker_init
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_timegroups_with_transform(
            val_ds,
            make_fm_minmax_transform(val_ds, apply_transf=True)), # apply_transf = True -> subtract fleetmedian
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=rng,
        worker_init_fn=_worker_init
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=collate_timegroups_with_transform(
            test_ds,
            make_fm_minmax_transform(test_ds, apply_transf=(True) )), # apply_transf = True -> subtract fleetmedian
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=rng,
        worker_init_fn=_worker_init
    )
       
    
    return train_loader, val_loader, test_loader


def load_feature_order()-> List[str]:
    return json.loads(Path(PATH_FEATURE_ORDER_FILE).read_text(encoding="utf-8"))

class DataTransformations:
    #_PATH_TAG = "_PATH"
    
    def __call__(self, x: torch.Tensor, metainf: Dict[str, Any]) -> torch.Tensor:
        
        if not hasattr(self, "transform"):
            raise RuntimeError("transform instanced does not exist, load first a transformation function")
        
        return self.transform(x,metainf)
    
    @classmethod
    def _build(cls,
               transform: Callable[[torch.Tensor, Dict[str,Any]], torch.Tensor]) -> "DataTransformations":
        
        instance = cls.__new__(cls)
        instance.transform = transform
        return instance
    
   
    @classmethod
    def batch_minmax(cls, feature_cols: List[str], dataset: MultiCSVDataset) -> Callable[[torch.Tensor, List[int]], torch.Tensor]:
        
        scalers: Dict[int, MinMaxScaler] = joblib.load(PATH_MINMAX_SCALER)
        feature_len = len(feature_cols)
        uid_to_pos: Dict[int,int] = dataset._uid_to_pos
        wt_arr: np.ndarray  = dataset._wt
        
        def _apply(X: torch.Tensor, uids: List[int]) -> torch.Tensor:
            
            if X.dim() != 2:
                raise ValueError(f"Expected shape of [TS x WT,F], got shape {tuple(X.shape)}")
            if X.size(0) != len(uids):
                raise ValueError(f"uids length = {len(uids)} != batch size {X.size(0)}")
            
            out_rows: List[torch.tensor] = []
            
            for i,uid in enumerate(uids):
                
                pos = uid_to_pos[int(uid)]
                wt_id = int(wt_arr[pos])
                scaler = scalers.get(wt_id)
                
                if scaler is None:
                    raise KeyError(f"Scaler with WT_ID = {wt_id} does not exist.")
                
                if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != feature_len:
                    raise ValueError(f"Feature length {feature_len} != scaler.n_features_in_ = {scaler.n_features_in_}.")
                
                row_np = X[i].detach().cpu().to(torch.float64).numpy().reshape(1, feature_len) # Shape (1, F)
                
                row_scaled = scaler.transform(row_np).ravel() # shape (F,)
                out_rows.append(torch.from_numpy(row_scaled))
            
            X_scaled = torch.stack(out_rows, dim=0).to(dtype=X.dtype, device=X.device)
            
            return X_scaled
        
        return _apply
    
    @classmethod
    def build_batch_minmax_uid_func(cls, loader: DataLoader ) -> Callable[[torch.Tensor, List[int]], torch.Tensor]:
        
        ds = loader.dataset
        for attr in ("_uid_to_pos", "_wt"):
            if not hasattr(ds, attr):
                raise AttributeError(f"{type(ds).__name__} attribute: {attr} is missing")
        feature_cols: List[str] = load_feature_order()
        
        return cls.batch_minmax(feature_cols=feature_cols, dataset=ds)
        
    
    @staticmethod
    @torch.no_grad()
    def substract_fleet_median_3d(X: torch.Tensor) -> torch.Tensor:
        """subtracts the fleet median from batch per timestamp

        Args:
            X (torch.Tensor): Shape [T, WT, F], T = timestamps, WT= Wind turbines, F= Features

        Returns:
            torch.Tensor: Shape [T, WT, F]
        """
        if X.dim() != 3:
            raise ValueError(f"Shape of X does not match, got shape {tuple(X.shape)}")
        
        # [T, 1, F]
        fm = torch.nanmedian(X, dim=1, keepdim=True).values 
        
        return X- fm
            

def loader_to_df_scaled(loader: DataLoader) -> pd.DataFrame:
    ds = cast(MultiCSVDataset, loader.dataset)
    Xbs, ts_list, wt_list = [], [], []
    
    for xb, uid in loader:
        Xbs.append(xb.cpu())
        pos = [ds._uid_to_pos[int(u)] for u in uid]
        ts_list += pd.to_datetime(ds._ts_ns[pos], unit="ns").tolist()
        wt_list += ds._wt[pos].tolist()
    
    X = torch.cat(Xbs).numpy()
    df_scaled = pd.DataFrame(X, columns=ds._signal_cols)
    df_scaled.insert(0, WT_ID, wt_list)
    df_scaled.insert(0, TS_COL, ts_list)
    return df_scaled
        
    
@dataclass
class TrainConfig:
    """ If you want to set the seed, use set_seed(...)
    """
    
    config_name: str
    #save_path = PATH_TRAINING_CFG
    
    # Training params
    batch_size: int = 18*14 # timestamps per batch
    epochs: int = 30
    lr: float = 0.001
    weight_decay : float = 0.0
    grad_clip_norm : None | float = 1.0
   
    
    # Early stopping params
    patience : int = 5
    min_delta: float = 1e-6
    
    # model architecture
    depth: int = None
    input_dim: int = len(load_feature_order())
    base_width: int = len(load_feature_order())      # size hidden layer 1
    width_decay: float = 0.5
    bottleneck_min : int = 2
    activation: str = "relu"
    dropout: float = 0.0
    leaky_relu_slope: Optional[float] = field(init=False, default = None)
    #current seed
    seed: int = field(init=False, default=32)
    # used for init
    base_seed: int = field(init=False, default=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Settings for cosine annealing warm restarts scheduler
    T0: int = 10 # cycle length in epochs
    Tmult: int = 2 # factor to extend cycle length
    eta_min_lower_bound: float = 1e-6
    
    val_split: float = 0.4
    #test_split: float = 0.2
    
    # num of layers for the encoder and decoder
    layer_depths: List[int] = field(default_factory=lambda: [2,3,4,5])
    
    # parameters to determine val and test set time periods
    available_start:    Optional[pd.Timestamp] = field(init=False, default=None)
    available_end:      Optional[pd.Timestamp] = field(init=False, default=None)
    min_gap: int = 4320 # gap in number of timestamps between val and test set
    
    # parameters to create the data loaders
    val_start_time:     Optional[pd.Timestamp] = field(init=False, default=None)
    val_end_time:       Optional[pd.Timestamp] = field(init=False, default=None)
    test_start_time:    Optional[pd.Timestamp] = field(init=False, default=None)
    test_end_time:      Optional[pd.Timestamp] = field(init=False, default=None)
    
    # signal order in table
    signal_cols:        Optional[List[str]] = field(init=False, default=None)
    
    # Name of the Date and Time col in table
    ts_col: str = TS_COL
    
    # Used for different restarts with different seeds
    n_restarts:int = 5
    
    #seed_list: Optional[List[int]] = field(init=False, default=None)
    seed_list: Optional[List[int]] = None
    
    #Part1 of the thesis
    part1: bool = False
    
    #Val-Phase: 2 val sets, one for hyperparameter tuning..
    # 2 for 2 disjoint valsets
    #val_phase: int = 1 # std
    choose_val_set: int = 1
    
    # dirty fix
    def set_seed(self,seed: int=32) -> None:
        self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if (self.device == "cuda") and torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        
    
    def __post_init__(self)-> None:
        self.signal_cols = load_feature_order()
        
        files = glob(os.path.join(PATH_IMPUTED, "*.csv"))
    
        times_all = get_unique_ts_ns(files, pd.to_datetime(TRAIN_END))
        
        self.available_start = times_all.min()
        self.available_end = times_all.max()
        
        
        val_start, val_end, test_start, test_end = pick_val_test_indices(
                                        times=pd.DatetimeIndex(times_all),
                                        config=self,
                                        choose_val_set=self.choose_val_set
                                        )
        self.val_start_time = val_start
        self.val_end_time = val_end
        self.test_start_time = test_start
        self.test_end_time = test_end
        
        rng = np.random.default_rng(self.seed)
        
        if self.seed_list is None:
            self.seed_list = [int(s) for s in rng.integers(0, 2**31-1, size=self.n_restarts)]
        
        self.set_seed(self.base_seed)
        
        print(json.dumps(asdict(self), indent=2, ensure_ascii=False,default=str))

def update_cfg(old_cfg: TrainConfig, new_cfg: TrainConfig) -> TrainConfig:
    """ Updates new cfg with new attributes with old settings.
        Attributes which will not be updated:
        - config_name
        - base_seed
        - available_start
        - available_end
        - min_gap
        - val_start_time
        - val_end_time
        - test_start_time
        - test_end_time
        - n_restarts
        - part1
        - val_phase
    """
    # new_cfg.config_name = old_cfg.config_name
    new_cfg.batch_size = old_cfg.batch_size
    new_cfg.epochs = old_cfg.epochs
    new_cfg.lr = old_cfg.lr
    new_cfg.weight_decay = old_cfg.weight_decay
    new_cfg.grad_clip_norm = old_cfg.grad_clip_norm
    new_cfg.patience = old_cfg.patience
    new_cfg.min_delta = old_cfg.min_delta
    new_cfg.depth = old_cfg.depth
    new_cfg.input_dim = old_cfg.input_dim
    new_cfg.base_width = old_cfg.base_width
    new_cfg.width_decay = old_cfg.width_decay
    new_cfg.bottleneck_min = old_cfg.bottleneck_min
    new_cfg.activation = old_cfg.activation
    new_cfg.dropout = old_cfg.dropout
    new_cfg.seed = old_cfg.seed
    new_cfg.leaky_relu_slope = getattr(old_cfg, "leaky_relu_slope", None)
    new_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    new_cfg.T0 = old_cfg.T0
    new_cfg.Tmult = old_cfg.Tmult
    new_cfg.eta_min_lower_bound = old_cfg.eta_min_lower_bound
    new_cfg.val_split = old_cfg.val_split
    #new_cfg.test_split = old_cfg.test_split
    new_cfg.layer_depths = old_cfg.layer_depths
    # new_cfg.available_start = old_cfg.available_start
    # new_cfg.available_end = old_cfg.available_end
    # new_cfg.min_gap = old_cfg.min_gap
    # new_cfg.val_start_time = old_cfg.val_start_time
    # new_cfg.val_end_time = old_cfg.val_end_time
    # new_cfg.test_start_time = old_cfg.test
    # new_cfg.test_end_time = old_cfg.
    new_cfg.signal_cols = old_cfg.signal_cols
    new_cfg.ts_col = old_cfg.ts_col
    # new_cfg.n_restarts = old_cfg.
    new_cfg.seed_list = old_cfg.seed_list
    # new_cfg.part1 = old_cfg.part1
    # new_cfg.val_phase = old_cfg.
    new_cfg.choose_val_set = getattr(old_cfg, "choose_val_set", 1)
    return new_cfg

def get_unique_ts_ns(files:List[str], train_end: pd.Timestamp, timedelta:str="10min")-> pd.DatetimeIndex:
    """Returns all unique timestamps after training end from all *.csv files.

    Args:
        files (List[str]): List of *.csv files
        train_end (pd.Timestamp): -
        timedelta (str): The first timestamp after train_end
    """
    ds_all = MultiCSVDataset(
        dataset_paths=files,
        signal_cols= load_feature_order(),
        start_time= pd.to_datetime(train_end) + pd.Timedelta(timedelta),
        ts_col=TS_COL
    )
    times_all = pd.to_datetime(np.unique(ds_all._ts_ns))
    return times_all

def get_n_of_train_timestamps()-> int:
    dfs = []
    ts_col = TS_COL
    
    for fp in Path(PATH_PC_FILTERING).glob("*.csv"):
        df = pd.read_csv(fp, usecols=[ts_col], parse_dates=[ts_col])
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)
    
    if not np.issubdtype(dfs[ts_col].dtype, np.datetime64):
        dfs[ts_col] = pd.to_datetime(dfs[ts_col], errors="raise")
    dfs = dfs[dfs[ts_col] >= pd.to_datetime(TRAIN_START)]
    dfs = dfs[dfs[ts_col] <= pd.to_datetime(TRAIN_END)]
    
    uniq_ts = pd.unique(dfs[ts_col])
    uniq_ts = pd.DatetimeIndex(uniq_ts, name=ts_col).sort_values()
    

    return int(len(uniq_ts))

def pick_val_test_indices(
    times: pd.DatetimeIndex,
    config: TrainConfig,
    train_end: str = TRAIN_END,
    choose_val_set: int = 1,
)-> Tuple[pd.Timestamp, pd.Timestamp,pd.Timestamp,pd.Timestamp]:
    """ Datasplit in timestamps for val and test set. Returns only the timestamps for start and end for val and test. This Function is used for the function build_dataloaders(...).
    The val sets will be picked from the first year after TRAIN_END. The test set will occupy all other timestamps in respect to the min_gap from the config.

    Args:
        times (pd.DatetimeIndex): DatetimeIndex of the available data excluding the train period
        config (TrainConfig): Some settings will be used for computations
        train_end (str): The date of the end of the training set
        val_set (int): val_set chooses which val set to be returned. Values for val_set are 1 or 2. These two val sets are disjoint in time and within the first year after train_end.

    Returns:
        (val_start, val_end, test_start, test_end): Timestamps
    """
    # if config.part1:
    #     return pd.to_datetime("2019-12-15 00:00:00"), pd.to_datetime("2020-04-01 00:00:00"),pd.to_datetime("2019-12-15 00:00:00"), pd.to_datetime("2020-04-01 00:00:00")
    
    if choose_val_set not in (1,2):
        raise ValueError(f"choose_valset can only be 1 or 2")
    
    val_frac = config.val_split
    
    min_gap = config.min_gap
    seed = config.base_seed
    available_start = config.available_start
    available_end = config.available_end
    
    if not isinstance(times, pd.DatetimeIndex):
        raise TypeError(f"input times must be from type pd.DatetimeIndex: found {type(times)}")
    if isinstance(train_end, str):
        train_end = pd.to_datetime(train_end)
    if isinstance(available_start,str):
        available_start = pd.to_datetime(available_start)
    if isinstance(available_end, str):
        available_end = pd.to_datetime(available_end)
    if (val_frac > 1) or (val_frac < 0):
        raise ValueError(f"check val_frac value: {val_frac}")
    #if (test_frac > 1) or (test_frac < 0):
    #    raise ValueError(f"check test_frac value: {test_frac}")
    if (available_start > available_end):
        raise ValueError(f"available_end: {available_end} need to be after available_start: {available_start}")
    if (available_start < train_end):
        raise ValueError(f"available_start:{available_start} cannot be before train_end:{train_end}")
    
    rng = np.random.default_rng(seed)
    STEP = pd.Timedelta(minutes=10)
    one_year_val_period_start= available_start + STEP
    one_year_val_period_end = one_year_val_period_start + pd.DateOffset(years=1)
    
    one_year_val_period = int((one_year_val_period_end - one_year_val_period_start) / STEP)
    valset_duration = max(1,int(np.floor(one_year_val_period*val_frac)))
    
    if (2 * valset_duration + min_gap > one_year_val_period):
        raise ValueError(f"There is not enough space in the first year after train period to pick the val set(s)")
    
    earliest_val1_start = one_year_val_period_start
    last_val1_start = one_year_val_period_end - STEP*(valset_duration*2-2 + min_gap +1)
    
    val1_window = int((last_val1_start - earliest_val1_start) / STEP)
    
    number = rng.integers(0,val1_window+1)
    
    chosen_val1_start = pd.Timestamp(earliest_val1_start + STEP*number)
    chosen_val1_end = pd.Timestamp(chosen_val1_start + STEP*(valset_duration-1))
    
    earliest_val2_start = pd.Timestamp(chosen_val1_end + STEP*(min_gap +1))
    last_val2_start = pd.Timestamp(one_year_val_period_end -STEP*(valset_duration-1))
    
    val2_window = int((last_val2_start - earliest_val2_start) / STEP)
    
    number2 = rng.integers(0, val2_window+1)
    
    chosen_val2_start = pd.Timestamp(earliest_val2_start + STEP*number2)
    chosen_val2_end = pd.Timestamp(chosen_val2_start + STEP*(valset_duration-1))
    
    if ((chosen_val1_end > one_year_val_period_end) or (chosen_val2_end > one_year_val_period_end) ):
        raise ValueError(f"Construction of val set time periods unsuccessful\n"
                         f"chosen_val1_end:{chosen_val1_end}, chosen_val2_end:{chosen_val2_end}\n"
                         f"1 Year val period last timestamp: {one_year_val_period_end}")
    
    test_start = chosen_val2_end + STEP*(min_gap+1)
    test_end = available_end
    
    if config.part1:
        val_start, val_end = chosen_val2_start, chosen_val2_end
        return val_start, val_end, val_start, val_end
    
    if choose_val_set == 1:
        val_start, val_end = chosen_val1_start, chosen_val1_end
    else:
        val_start, val_end = chosen_val2_start, chosen_val2_end
    
    
    
    return val_start, val_end, test_start, test_end
    

def _dict_to_groups_list(dicti: dict[int, List[int]], 
                         min_group_size: int = 14 # Penmanshiel
                         ) -> List[List[int]]:
    keys_sorted = sorted(dicti.keys())
    return [dicti[k] for k in keys_sorted if len(dicti[k]) >= min_group_size]

def rebuild_grouped_loader(loader: DataLoader,
                           seed:int,
                           shuffle: bool,
                           batch_size:int
                           )-> DataLoader:
    """Used to faster reload the dataloader. This Function is only for reproducing experiments.

    Args:
        loader (DataLoader): ...
        seed (int): extract the seed from TrainConfig
        shuffle (bool): if needed.
        batch_size (int): extract the batch_size from TrainConfig

    Returns:
        DataLoader: Reinitialized DataLoader to reproduce experiments.
    """
    
    ds = loader.dataset
    groups_dict = make_time_groups_from_dataset(ds)
    groups_list = _dict_to_groups_list(groups_dict)
    
    sampler = GroupedBatchSampler(groups_list=groups_list,
                                  timestamps_per_batch=batch_size,
                                  shuffle=shuffle,
                                  seed=seed)
    
    rng = torch.Generator()
    rng.manual_seed(seed)
    
    return DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        generator=rng
        
    )
    

    