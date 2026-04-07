from __future__ import annotations
from enum import StrEnum
from helperfunctions import helper as owdl
from helperfunctions.helper import TrainConfig
from helperfunctions.intern_constants import PATH_TO_BEST_MODEL_DIR, BEST_MODEL, MEAN_LOSS_PER_SAMPLE, TS_COL, WT_ID, RE_PREFIX, PATH_PROJECT_ROOT
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Callable, Optional, Sequence, TypedDict, Any, Dict, List
import copy
import numpy as np
import os
import pandas as pd
import torch
import sys
from contextlib import nullcontext

# The following is used for training_lib.train_with_early_stopping
class HistoryKeys(StrEnum):
    EPOCH = "epoch"
    TRAIN_MEAN_EPOCH= "train_mean_epoch"
    VAL_MEAN_EPOCH= "val_mean_epoch"
    TRAIN_CONFIG= "train_config"

class HistoryDict(TypedDict):
    epoch: int
    train_mean_epoch: float
    val_mean_epoch: float
    train_config: Any

class EvalKeys(StrEnum):
    HISTORY= "history"
    BEST_VAL= "best_val"
    MODEL_PARAMS= "model_params"
    MODEL_NAME= "model_name"
    DEPTH= "depth"
    DIR_PATH= "dir_path"
# A Return Value of training_lib.train_with_early_stopping
class EvalDict(TypedDict):
    history: List[HistoryDict]
    best_val: float
    model_params: int
    model_name: str
    depth: int
    dir_path: Path

class CheckpointKeys(StrEnum):
    MAX_EPOCH= "max_epoch"
    MODEL_STATE_DICT= "model_state_dict"
    OPTIMIZER_STATE_DICT= "optimizer_state_dict"
    TRAIN_CONFIG= "train_config"
    BEST_VAL= "best_val"
    FILE_PATH= "file_path"
    DIR_PATH= "dir_path"
    TRAIN_MEAN_EPOCH= "train_mean_epoch"
    #VAL_MEAN_EPOCH= "val_mean_epoch" #redundant - best_val same 
    MODEL_PARAMS= "model_params"
    SCHEDULER_STATE_DICT= "scheduler_state_dict"
    SCALER_STATE_DICT = "scaler_state_dict"
            
class CheckpointDict(TypedDict):
    max_epoch: int
    model_state_dict: Dict[str,Any]
    optimizer_state_dict: Dict[str,Any]
    scheduler_state_dict: Dict[str,Any]
    train_config: TrainConfig
    best_val: float
    file_path: Path
    dir_path: Path
    train_mean_epoch: float
    #val_mean_epoch: float
    model_params: int
    scaler_state_dict : Dict[str, Any]
    
def get_model_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#def get_device() -> torch.device:
#    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_activation(name: str, 
                   leaky_relu_slope: float = 0.1, 
                   ) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_slope)
    else:
        raise ValueError(f"Unknown activation: {name}")

def linear_segment(feats_in: int, feats_out: int, activation: nn.Module, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(feats_in, feats_out), nn.BatchNorm1d(feats_out)]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(activation)
    
    return nn.Sequential(*layers)

def build_enc_dec_layers(input_dim: int, 
                         depth: int, 
                         base_width: int, 
                         width_decay: float, 
                         bottleneck_min: int, 
                         activation: str, 
                         dropout: float,
                         leaky_relu_slope: float = 0.1,
                         ) -> Tuple[nn.Sequential, nn.Sequential, int, list[int]]:
    """
        If you change this code,  max_depth_for_width_decay must be adjusted too.
    """
    widths = []
    new_base_width = base_width
    
    # [(in,out),...]
    encoder_dims: List[Tuple[int,int]] = []
    
    input = input_dim
    act = get_activation(activation,
                         leaky_relu_slope=leaky_relu_slope,
                         )
    for _ in range(depth):
        if len(widths) == 0: # prevent layer 1 to be same as layer 0
            new_base_width *= width_decay
        
        candidate = int(np.floor(new_base_width))
        if bottleneck_min > candidate:
            break        
        
        if len(widths)>0:
            prev = widths[-1]
            if candidate >= prev:
                candidate = prev -1
            
            if candidate < bottleneck_min:
                break

        widths.append(candidate)
        encoder_dims.append((input, candidate))
        input = candidate
        
        new_base_width *=width_decay
    
    if not encoder_dims:
        raise ValueError(f"Check bottleneck dimension {bottleneck_min} and base width {base_width}") 
            
    bottleneck = widths[-1]
    
    encoder_layers = [ linear_segment(input, output,act, dropout) for input, output in encoder_dims ]
    
    decoder_dims = [ (output, input) for input, output in reversed(encoder_dims)]
    
    decoder_layers: list[nn.Module] = []
    
    for input, output in decoder_dims[:-1]:
        decoder_layers.append(linear_segment(input, output, act, dropout))
    
    last_in, last_out = decoder_dims[-1]
    decoder_layers.append(linear_segment(last_in, last_out, act, dropout))
    
    return (nn.Sequential(*encoder_layers),
            nn.Sequential(*decoder_layers),
            bottleneck,
            widths)
    

class Autoencoder(nn.Module):
    def __init__(self,
                 cfg:owdl.TrainConfig,
                 #input_dim: int,
                 #depth: int,
                 #base_width: int,
                 #width_decay: float,
                 #bottleneck_min: int,
                 #activation: str= "relu",
                 #dropout: float = 0.0
                 ):
        super().__init__()
        self._activation_name = getattr(cfg, "activation", "relu")
        slope = getattr(cfg, "leaky_relu_slope", None)
        if slope is None:
            act = str(self._activation_name).lower()
            slope = 0.1 if act == "leaky_relu" else 0.0
        self._leaky_relu_slope = float(slope)
        
        self.encoder, self.decoder, _,_ = build_enc_dec_layers(
            input_dim=cfg.input_dim,
            depth= cfg.depth,
            base_width=cfg.base_width,
            width_decay=cfg.width_decay,
            bottleneck_min=cfg.bottleneck_min,
            activation=cfg.activation,
            dropout=cfg.dropout,
            leaky_relu_slope=self._leaky_relu_slope,
        )
        self.apply(self._init_he_to_corresponding_activation)
    
    def _init_he_to_corresponding_activation(self, module: nn.Module):
        name = self._activation_name.lower()
        if isinstance(module, nn.Linear):
            if name == "leaky_relu":
                a = self._leaky_relu_slope
            else: # relu
                a = 0.0
            # HE-init
            nn.init.kaiming_normal_(module.weight, a=a, mode="fan_in", nonlinearity=name)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def check_dir(path:Path) -> Path:
    path.mkdir(exist_ok=True)
    return path

class EarlyStopping:
    
    def __init__(self, min_delta: float = 1e-4, 
                 patience: int = 15, 
                 ):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = np.inf
        self.no_improve_epochs: int = 0
    
    def step(self, re: float) -> Tuple[bool,bool]:
        improved = ( self.best == np.inf) or  (self.best - re) > self.min_delta
        if improved:
            self.best = re
            self.no_improve_epochs = 0
            stop = False
        else:
            self.no_improve_epochs +=1
            stop = self.no_improve_epochs > self.patience
        return stop,improved

def _get_autocast_ctx(device: torch.device |str, scaler: Optional[torch.cuda.amp.GradScaler]):
    if scaler is None:
        return nullcontext()
    
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
    
    if device_type != "cuda" or not torch.cuda.is_available():
        return nullcontext()
    
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(dtype=torch.float16)
    
    return nullcontext()


def train_a_epoch(model: nn.Module,
                  dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  *,
                  loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  scaler: Optional[torch.cuda.amp.GradScaler] = None,
                  grad_clip_norm: Optional[float] = None,
                  scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts] = None,
                  curr_epoch: Optional[int] = None,
                  ) -> Tuple[float,float]:
    """ Make sure loss_fn returns losses per features

    Returns:
        Tuple[float,List[List[float]]]: mean_epoch, per_sample_loss_list
    """
    batch_loss_list: List[float] = []
    per_sample_loss_list: List[List[float]] = []
     
    model.train()
    num_batches = len(dataloader)
    
    for batch_idx ,(x, _) in enumerate(dataloader):
        
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        
        if scaler is not None:
            
            ctx  = _get_autocast_ctx(device=device, scaler=scaler)
            with ctx:

                y_hat = model(x)
                # we want to get the loss per sample.
                loss_elems = loss_fn(y_hat,x)
                loss_per_sample = loss_elems.mean(dim=1)
                loss = loss_per_sample.mean()
                
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        
        else:
            y_hat = model(x)
            loss_elems = loss_fn(y_hat,x)
            loss_per_sample = loss_elems.mean(dim=1)
            loss = loss_per_sample.mean()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
        
        if scheduler is not None and curr_epoch is not None:
            progress = (batch_idx +1) / num_batches # (0,1]
            scheduler.step((curr_epoch - 1)+progress)
        
        batch_loss_list.append(loss.detach().item())
        per_sample_loss_list.append(loss_per_sample.detach().cpu().tolist())
    mean_epoch = sum(batch_loss_list) / len(batch_loss_list)
            
    return mean_epoch, per_sample_loss_list

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    return_uids: bool = False
) -> Tuple[float, List[List[float]]] | Tuple[float, List[List[float]],List[List[int]]]:
    """Evaluate the model on the given loader.
    
    Returns:
        (mean_loss, elem_loss_list, uids_batches) 
        OR
        (mean_loss, elem_loss_list)
    """
    model.eval()
    
    total_loss = 0.0
    n_samples = 0
    elem_loss_list: List[np.ndarray] = []
    uids_batches: List[np.ndarray] = [] if return_uids else None
    
    with torch.inference_mode():
        
        for x, uids_flat in loader:
            
            x = x.to(device, non_blocking=True)
            y_hat = model(x)
            elem_loss = loss_fn(y_hat, x) # loss_fn must return a element-wise error 
            loss_per_sample = elem_loss.mean(dim=1)
            total_loss += loss_per_sample.sum().item()
            n_samples += loss_per_sample.numel()
            
            elem_loss_list.append(elem_loss.detach().cpu().numpy())
            if return_uids:
                uids_batches.append(np.asarray(uids_flat, dtype=np.int64))
    mean_loss = (total_loss / n_samples)
    if return_uids:
        return mean_loss, elem_loss_list, uids_batches
    
    return mean_loss, elem_loss_list

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: owdl.TrainConfig,
    *,
    es: EarlyStopping,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    filename_prefix: str = "",
    #max_epochs: int = 5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    #grad_clip_norm: Optional[float] = None,
    use_lr_scheduler: bool = False,
) -> Tuple[EvalDict, nn.Module]:
    """ Training with early stopping.
        Make sure to initialize the EarlyStopping class.
        The loss function should return the losses element-wise/feature-wise e.g. torch.nn.MSELoss(reduction=none)
        The best performing model will be saved at Directory: PATH_TO_BEST_MODEL_DIR (see intern_constants.py)
    
    Returns:
        Tuple[EvalDict, nn.Module]
        
    """
    def _arrow(curr:float, prev:float | None, tol:float = 0.0) -> str:
        if prev is None:
            return "•"
        d = curr - prev
        if d < -tol:
            return "↓" # improvement
        if d > tol:
            return "↑" #deterioration
        else: 
            return "→" # same
        
    check_dir(PATH_TO_BEST_MODEL_DIR)
    
    history: List[HistoryDict] = []
    #TODO
    # eval und history are not set for return value
    eval: EvalDict = {
        EvalKeys.HISTORY: history,
        EvalKeys.BEST_VAL: np.inf,
        EvalKeys.MODEL_PARAMS: get_model_params(model=model),
        EvalKeys.MODEL_NAME: "",
        EvalKeys.DEPTH: None,
        EvalKeys.DIR_PATH: "",
    }
    
    filename = filename_prefix +".pth"  
    
    scheduler = None
    if use_lr_scheduler:
        # set up warm restarts scheduler
        base_lr = config.lr
        T0 = config.T0
        Tmult = config.Tmult
        eta_min = max(base_lr*0.1, config.eta_min_lower_bound)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0= T0,
            T_mult= Tmult,
            eta_min= eta_min,
        )
    
    pbar = tqdm(range(1, config.epochs +1), 
                desc="Epochs", 
                unit="ep", 
                dynamic_ncols=True,
                leave=True,
                disable=False,
                file=sys.stdout)
    
    prev_train: float | None = None
    prev_val: float | None = None
    prev_best: float | None = None
    
    for epoch in pbar:
        
        train_mean_loss, _ = train_a_epoch(model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=config.device,
            loss_fn=loss_fn,
            scaler=scaler,
            grad_clip_norm=config.grad_clip_norm,
            scheduler=scheduler,
            curr_epoch=epoch)
        
        val_mean_loss, _ = validate_epoch(
            model=model, 
            loader=val_loader,
            device=config.device,
            loss_fn=loss_fn
        )
        
        # may not have sufficient RAM -
        
        history.append({
            HistoryKeys.EPOCH: epoch,
            #re per epoch
            HistoryKeys.TRAIN_MEAN_EPOCH: train_mean_loss,
            HistoryKeys.VAL_MEAN_EPOCH: val_mean_loss,
            #loss per sample per batch (elem loss)
            #"train_ps_loss": train_ps_loss,
            #"val_ps_loss": val_ps_loss,
            #HistoryKeys.TRAIN_CONFIG: config,
        })
        tol = getattr(es, "min_delta", 0.0)
        arrow_train = _arrow(train_mean_loss, prev_train, tol)
        arrow_val = _arrow(val_mean_loss, prev_val, tol)
        arrow_best = _arrow(es.best, prev_best, 0.0)
        
        stop, improved = es.step(val_mean_loss)
        
        lr = (scheduler.get_last_lr()[0] if scheduler is not None
              else optimizer.param_groups[0]["lr"])
        
        pbar.set_postfix(
            train=f"{arrow_train} {train_mean_loss:.4f}",
            val=f"{arrow_val} {val_mean_loss:.4f}",
            best=f"{arrow_best} {es.best:.4f}",
            #improved=f"{improved}",
            lr=f"{lr:.2e}",
            refresh=True,
        )
        prev_train = train_mean_loss
        prev_val = val_mean_loss
        prev_best = es.best
        
        if improved:
            
            dir = PATH_TO_BEST_MODEL_DIR / f"{filename_prefix}"
            fpath = (dir / filename)
            eval["best_val"] = es.best
            eval["dir_path"] = dir
            checkpoint: CheckpointDict = {
                CheckpointKeys.MAX_EPOCH: config.epochs,
                CheckpointKeys.MODEL_STATE_DICT: copy.deepcopy(model.state_dict()),
                CheckpointKeys.OPTIMIZER_STATE_DICT: copy.deepcopy(optimizer.state_dict()),
                CheckpointKeys.SCHEDULER_STATE_DICT: copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                CheckpointKeys.TRAIN_CONFIG: config,
                CheckpointKeys.BEST_VAL: es.best,
                CheckpointKeys.FILE_PATH: fpath,
                CheckpointKeys.DIR_PATH: dir,
                CheckpointKeys.TRAIN_MEAN_EPOCH: train_mean_loss,
                #CheckpointKeys.VAL_MEAN_EPOCH: val_mean_loss, # redundant parameter - see best_val
                CheckpointKeys.MODEL_PARAMS: get_model_params(model=model),
                CheckpointKeys.SCALER_STATE_DICT: (scaler.state_dict() if scaler else None),
                
                
            }
            os.makedirs(dir, exist_ok=True)
            torch.save(checkpoint, fpath )
        if stop:
            break
        
    eval["history"] = history
    eval["model_name"] = filename.removesuffix(".pth")
    eval["depth"] = config.depth
    
    return eval, model


def load_autoencoder(
    device: torch.device,
    dir: Path | str = Path(PATH_TO_BEST_MODEL_DIR / BEST_MODEL),
    )-> Tuple[Autoencoder, torch.optim.Optimizer, CheckpointDict, Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts], torch.amp.GradScaler | None]:
    
    ckpt:CheckpointDict = torch.load( dir, map_location=device, weights_only=False)
    
    ckpt_path = Path(dir)

    orig_dir = Path(ckpt["dir_path"]) if "dir_path" in ckpt else ckpt_path.parent
    orig_file = Path(ckpt["file_path"]) if "file_path" in ckpt else ckpt.path

    #run_name = orig_dir.name

    new_dir = PATH_TO_BEST_MODEL_DIR / orig_dir.name
    new_file = new_dir / orig_file.name

    new_dir.mkdir(parents=True, exist_ok=True)

    #fix
    ckpt["dir_path"] = new_dir
    ckpt["file_path"] = new_file

    cfg: owdl.TrainConfig = ckpt["train_config"]
    
    if not hasattr(cfg, "leaky_relu_slope") or cfg.leaky_relu_slope is None:
        act = str(getattr(cfg, "activation", "relu")).lower()
        cfg.leaky_relu_slope = 0.1 if act == "leaky_relu" else 0.0
    
    model = Autoencoder(cfg=cfg).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)
    # if not return_scheduler: 
    #     return model, optimizer, ckpt
    
    T0 = cfg.T0
    Tmult = cfg.Tmult
    eta_min = max(cfg.lr * 0.01, cfg.eta_min_lower_bound)
    
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts] = None
    if (CheckpointKeys.SCHEDULER_STATE_DICT in ckpt 
        and ckpt["scheduler_state_dict"] is not None):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = T0,
            T_mult = Tmult,
            eta_min = eta_min
        )
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
    scaler = None
    if ckpt["scaler_state_dict"]:
        scaler = torch.amp.GradScaler()
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    
    return model, optimizer, ckpt, scheduler, scaler
    
    

@torch.no_grad()
def uids_to_wt_ts(
    uid_batches: Sequence[Sequence[int]],
    elem_losses: Sequence[np.ndarray], 
    loader: DataLoader[tuple[torch.Tensor, list[int]]]
    ) -> pd.DataFrame:
    
    uids = np.concatenate(uid_batches, axis=0)
    errs = np.concatenate(elem_losses, axis=0)
    
    ds:owdl.MultiCSVDataset = loader.dataset
    
    
    pos = np.fromiter((ds._uid_to_pos[int(uid)] for uid in uids), dtype=np.int64, count=uids.size)
    
    ts_ns = pd.to_datetime(ds._ts_ns[pos])
    wt = ds._wt[pos].astype(int)
    
    df  = pd.DataFrame({
        "uid": uids,
        TS_COL: ts_ns,
        WT_ID: wt
    })
    loss_per_sample = errs.mean(axis=1).astype(float)
    df[MEAN_LOSS_PER_SAMPLE]= loss_per_sample
    #pos = df.columns.get_loc(WT_ID) +1
    #df.insert(pos, "Loss_per_sample", loss_per_sample)
    names = list(owdl.load_feature_order())
    
    for i in range(errs.shape[1]):
        df[f"{RE_PREFIX}{names[i]}"] = errs[:,i].astype(float)
    
    re_cols = [f"{RE_PREFIX}{n}" for n in names]
    order = ["uid", TS_COL, WT_ID, MEAN_LOSS_PER_SAMPLE, *re_cols]
    df = df.reindex(columns=order)
    
    return df

def clean_train_by_quantile(
    df: pd.DataFrame,
    df_RE: pd.DataFrame,
    q: float = 0.995,
    *,
    wt_col: str = WT_ID,
    ts_col: str = TS_COL,
    re_col: str = MEAN_LOSS_PER_SAMPLE,
    show_dropped: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Uses the output from training_lib.uids_to_wt_ts() to apply quantile method to filter the timestamps on df. The method returns the filtered training table

    Args:
        df_RE (pd.DataFrame): needs to be the output from training_lib.uids_to_wt_ts() otherwise the operation will fail.
        q (float, optional): q-value. Defaults to 0.995.
        wt_col (str, optional): The name of the column that represents the WT ID. Defaults to WT_ID.
        ts_col (str, optional): Column of timestamps. Defaults to TS_COL.
        re_col (str, optional): Column of reconstruction errors. Defaults to MEAN_LOSS_PER_SAMPLE.
        show_dropped (bool, optional): show the dropped ts - wt pairs.
    Returns:
        df_filtered : the filtered df after applying quantile method (show_dropped = False)
        (df_filtered, flagged): Dataframe that contains wt_col and ts_col, values that have been thrown, if show_dropped is True.
    """
   
    thr_val = df_RE.groupby(wt_col)[re_col].quantile(q)
    
    # 1D Series: index= WT ID, value= quantile
    flagged = df_RE[wt_col].map(thr_val)
    
    if flagged.isna().any():
        missing = df_RE.loc[flagged.isna(), wt_col].unique().tolist()
        raise KeyError(f" Missing threshold for WT_IDs : {missing}")
    
    flagged = df_RE.loc[df_RE[re_col] > flagged, [wt_col, ts_col]].copy()
    
    if flagged.isna().values.any():
        raise ValueError("flagged contains NaN entries.")
    if flagged.duplicated([wt_col, ts_col]).any():
        dupli = (
            flagged[flagged.duplicated([wt_col, ts_col], keep=False)]
            .sort_values([wt_col, ts_col]).head(10)
            )
        raise ValueError(f"Duplicates found in flagged. Examples: \n{dupli}")
    
    flagged[ts_col] = pd.to_datetime(flagged[ts_col], errors="raise")
    df= df.copy()
   
    df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
    flagged[wt_col] = flagged[wt_col].astype(df[wt_col].dtype)
    
    df_filtered = (
        df.merge(flagged.assign(_drop=1), on=[wt_col,ts_col], how="left")
        .query("_drop != 1")
        .drop(columns="_drop")
    )
    
    
    return (df_filtered, flagged) if show_dropped else df_filtered
    
def eval_model(
            model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            )-> pd.DataFrame:
    """Evaluates the model on given data and returning df table containing the signal-wise- and mean loss  per wind turbine.

    Make sure the loss_fn returns element-wise losses.

    Returns:
        pd.DataFrame: Table of losses
    """
    model_device = next(model.parameters()).device

    if isinstance(device, str):
        device = torch.device(device)
    if device != model_device:
        device = model_device

    _, elem_losses, uid_batches = validate_epoch(
            model=model,
            loader=data_loader,
            device=device,
            loss_fn=loss_fn,
            return_uids=True
            )
    df_eval = uids_to_wt_ts(
            uid_batches=uid_batches,
            elem_losses=elem_losses,
            loader=data_loader,
            )
    
    return df_eval

def get_model_results(src: Path,   
                      best_n: int = 5,
                      report_min_max: bool = False
                    )-> List[ Tuple[Autoencoder, torch.optim.Optimizer, CheckpointDict, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts | Any, torch.amp.GradScaler | None]]:
    """Prints early stopping results from best_n models found in PATH_TO_BEST_MODEL_DIR

    Args:
        src (Path): The directory containing the models

    Returns:
        List[Autoencoder, torch.optim.Optimizer, CheckpointDict, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts | Any, torch.amp.GradScaler | None]: Ascending list per val error of best models.
        optional: reports in addition (min,max) of all validation errors.
    """
    src = Path(src)
    if not os.path.exists(src):
        raise ValueError(f"source: {src} does not exist.")
    
    files = sorted(src.rglob("*.pth"))
    
    if len(files) == 0:
        raise ValueError(f"No files found in src")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ae_list = [load_autoencoder(device, f) for f in files]
    
    ae_list.sort(key=lambda e: e[2]["best_val"])
    
    print_list = [ckpt for (_,_,ckpt,_,_) in ae_list ]
    print_list = print_list[:best_n]
    print(*(
        f"name:{os.path.basename(evdict['file_path'])}\n best val error:{evdict['best_val']}\n best train error:{evdict['train_mean_epoch']}\n model params:{evdict['model_params']}\n"
        
        for evdict in print_list),sep="\n")
    
    if report_min_max and len(ae_list)>1:
        min = ae_list[0][2]["best_val"]
        max = ae_list[-1][2]["best_val"]
        return ae_list[:best_n],(min,max)
    
    return ae_list[:best_n]

def max_depth_for_width_decay(
    base_width: int,
    width_decay: float,
    bottleneck_min: int,
) -> Tuple[int, List[int]]:
    """Determines the maximal width for the number of layers of the decoder/Encoder

        This code adapts the logic from build_enc_dec_layers
    Args:
        see helperfunctions.helper.Trainconfig

    Returns:
        List[int]: width of max. layers N, starting from [2 .. N]
    """
    
    if base_width <= 0 or width_decay <= 0 or bottleneck_min <= 0:
        raise ValueError(f"base_width, width_decay, bottleneck_min mus be > 0")
    
    if width_decay >= 1:
        raise ValueError(f"width_decay must be < 1")

    widths: List[int] = []
    new_base_width: float = float(base_width)
    
    while True:
        if len(widths) == 0:
            new_base_width *= width_decay
        
        candidate = int(np.floor(new_base_width))
        if bottleneck_min > candidate:
            break

        if len(widths)>0:
            prev = widths[-1]
            if candidate >= prev:
                candidate = prev -1
            
            if candidate < bottleneck_min:
                break
    
        widths.append(candidate)
        new_base_width *=width_decay
    
    return list(range(1, len(widths) + 1))

def make_path_relative(path: str) -> str:
    
    proj_path = Path(PATH_PROJECT_ROOT).resolve()
    p = Path(path).resolve()

    try:
        return p.relative_to(proj_path).as_posix()
    except ValueError:
        repo_name = proj_path.name
        parts = p.parts
        if repo_name in parts:
            rel = Path(*parts[parts.index(repo_name)+1 :])
            return rel.as_posix()
    
        raise ValueError(f"Path is not inside project root.")