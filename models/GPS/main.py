import time
import numpy as np
import torch
from dataclasses import replace

from .config import (
    TrainingConfig, WEIGHTS_DIR, WEIGHTS_CPC_BEST_DIR, device,
    ORIGIN_BATCH_SIZE, DEST_BATCH_SIZE, NAN_BATCH_THRESHOLD,
    save_metrics_to_csv, save_model_weights,
)
from .model import make_model
from .loss import compute_loss_for_city
from .metrics import compute_metrics, evaluate_full_matrix
from .data_load import prepare_single_city_data, prepare_multi_city_data
from models.shared.plotting import save_loss_plot


# ─── Unified training loop ───────────────────────────────────────────────────

def _train_loop(run_id, run_name, config, model, city_datas,
                is_multi=False, train_city_ids=None, val_city_ids=None,
                test_city_ids=None):
    """
    Unified training loop for single-city and multi-city modes.
    city_datas: dict {city_id: city_data}
    """
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Diagnostic: verify config actually differs between runs
    import hashlib
    w_hash = hashlib.md5(
        b''.join(p.data.cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()[:8]
    print(f"  [diag] encoder={config.encoder_type} decoder={config.decoder_type} "
          f"pe={config.pe_type} norm={config.gps_norm_type} loss={config.loss_type}")
    print(f"  [diag] init_weights_hash={w_hash}")
    # Check PE data
    sample_cd = list(city_datas.values())[0]
    gd = sample_cd.get('graph_data')
    if gd is not None and hasattr(gd, 'pe') and gd.pe is not None:
        print(f"  [diag] pe.shape={gd.pe.shape}  pe[:3,:3]={gd.pe[:3,:3].tolist()}")
    elif gd is not None:
        print(f"  [diag] pe=None (no positional encoding)")
    od_train = sample_cd.get('od_matrix_train')
    if od_train is not None:
        print(f"  [diag] od_train_hash={hashlib.md5(np.ascontiguousarray(od_train).tobytes()).hexdigest()[:8]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

    max_epochs = config.mc_epochs if is_multi else config.epochs
    history = {'train_loss': [], 'val_loss': [], 'val_cpc_full': [], 'val_cpc_nz': []}
    best_val_loss = float('inf')
    patience_count = 0
    best_state = None

    # Split cities into train / val
    if is_multi:
        assert train_city_ids is not None and val_city_ids is not None
        train_cds = {cid: city_datas[cid] for cid in train_city_ids if cid in city_datas}
        val_cds = {cid: city_datas[cid] for cid in val_city_ids if cid in city_datas}
        if test_city_ids is None:
            seen = set(train_city_ids) | set(val_city_ids)
            test_city_ids = [cid for cid in city_datas if cid not in seen]
    else:
        cid = list(city_datas.keys())[0]
        cd = city_datas[cid]
        train_cds = {cid: cd}
        val_cd = dict(cd)
        val_cd['od_matrix_train'] = cd['od_matrix_np'] * cd['val_mask']
        val_cd['outflow_train'] = cd['outflow_val']
        val_cd['inflow_train'] = cd['inflow_val']
        val_cd['nonzero_dest_dict'] = cd['val_dest_dict']
        val_cds = {cid: val_cd}

    epoch = 0
    status = 'ok'
    loss_plot_path = WEIGHTS_DIR.parent / "loss_plots" / f"{run_id}_loss.png"
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        city_ids_shuffled = list(train_cds.keys())
        np.random.shuffle(city_ids_shuffled)
        epoch_losses = []
        nan_count = 0
        total_batches = 0

        for cid in city_ids_shuffled:
            cd = train_cds[cid]
            cc = replace(config, n_dest_sample=cd.get('city_n_dest', config.n_dest_sample)) if is_multi else config
            origins = np.array(list(cd['nonzero_dest_dict'].keys()))
            np.random.shuffle(origins)
            for bs in range(0, len(origins), ORIGIN_BATCH_SIZE):
                batch = origins[bs:bs + ORIGIN_BATCH_SIZE].tolist()
                optimizer.zero_grad()
                loss = compute_loss_for_city(model, cd, cc, origin_batch_indices=batch)
                total_batches += 1
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        # Divergence check
        if total_batches > 0 and nan_count / total_batches > NAN_BATCH_THRESHOLD:
            print(f"  NaN divergence @ epoch {epoch} ({nan_count}/{total_batches} NaN batches)")
            status = 'nan_diverged'
            break

        train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')

        # Validation: loss + CPC on full matrix
        model.eval()
        val_losses = []
        val_cpc_fulls = []
        val_cpc_nzs = []
        with torch.no_grad():
            for vcid, vcd in val_cds.items():
                vl = compute_loss_for_city(model, vcd, config)
                if not (torch.isnan(vl) or torch.isinf(vl)):
                    val_losses.append(vl.item())
                eval_cd = city_datas[vcid] if is_multi else list(city_datas.values())[0]
                _, mf, mnz = evaluate_full_matrix(model, eval_cd, config, dest_batch_size=DEST_BATCH_SIZE)
                val_cpc_fulls.append(mf['CPC'])
                val_cpc_nzs.append(mnz['CPC'])

        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        avg_cpc_full = np.mean(val_cpc_fulls)
        avg_cpc_nz = np.mean(val_cpc_nzs)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_cpc_full'].append(avg_cpc_full)
        history['val_cpc_nz'].append(avg_cpc_nz)

        if not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            flag = ' *'
        else:
            patience_count += 1
            flag = ''

        if epoch % 5 == 0 or epoch == 1:
            nan_str = f" NaN:{nan_count}" if nan_count > 0 else ""
            print(f"  {epoch:3d}/{max_epochs}  train={train_loss:.4f}  val={avg_val_loss:.4f}  "
                  f"CPC_full={avg_cpc_full:.4f}  CPC_nz={avg_cpc_nz:.4f}  "
                  f"{time.time()-t0:.1f}s{flag}{nan_str}")
        if patience_count >= config.patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    saved_plot_path = save_loss_plot(
        history['train_loss'],
        history['val_loss'],
        title=f"{run_name} Loss",
        save_path=loss_plot_path,
    )
    if saved_plot_path is not None:
        print(f"  -> Loss plot saved to {saved_plot_path}")
        model.loss_plot_path = str(saved_plot_path)

    if status == 'nan_diverged':
        dummy = {'CPC': 0.0, 'MAE': float('inf'), 'RMSE': float('inf')}
        save_metrics_to_csv(run_id, run_name, config, dummy, dummy, dummy, n_params, epoch, status)
        return {
            'name': run_name, 'model': model, 'config': config, 'history': history,
            'metrics_full': dummy, 'metrics_nonzero': dummy, 'metrics_test_pairs': dummy,
            'pred_matrix': None, 'status': status,
            'loss_plot_path': str(saved_plot_path) if saved_plot_path is not None else None,
        }

    # Evaluate on last and best weights
    if is_multi:
        full_eval_cities = city_datas
        test_eval_cities = (
            {cid: city_datas[cid] for cid in test_city_ids if cid in city_datas}
            if test_city_ids is not None else {}
        )
        if not test_eval_cities:
            test_eval_cities = full_eval_cities
    else:
        full_eval_cities = {list(city_datas.keys())[0]: list(city_datas.values())[0]}
        test_eval_cities = full_eval_cities

    def eval_all(label):
        all_mf = []
        all_mnz = []
        all_mt = []
        per_city = []
        test_eval_ids = set(test_eval_cities)
        for ecid, ecd in full_eval_cities.items():
            pred, mf, mnz = evaluate_full_matrix(model, ecd, config, dest_batch_size=DEST_BATCH_SIZE)
            mt = None
            if is_multi:
                if ecid in test_eval_ids:
                    mt = {'CPC': mf['CPC'], 'MAE': mf['MAE'], 'RMSE': mf['RMSE']}
                    all_mt.append(mt)
            else:
                mt = compute_metrics(pred[ecd['test_mask']], ecd['od_matrix_np'][ecd['test_mask']].astype(float))
                all_mt.append(mt)
            all_mf.append(mf)
            all_mnz.append(mnz)
            per_city.append({
                'city_id': ecid, 'CPC_full': mf['CPC'], 'CPC_nz': mnz['CPC'],
                'MAE': mf['MAE'], 'RMSE': mf['RMSE'],
                'CPC_test': mt['CPC'] if mt is not None else float('nan'),
            })
        avg_mf = {k: np.mean([m[k] for m in all_mf]) for k in all_mf[0]}
        avg_mnz = {k: np.mean([m[k] for m in all_mnz]) for k in all_mnz[0]}
        avg_mt = (
            {k: np.mean([m[k] for m in all_mt]) for k in all_mt[0]}
            if all_mt else
            {'CPC': avg_mf['CPC'], 'MAE': avg_mf['MAE'], 'RMSE': avg_mf['RMSE']}
        )
        print(f"\n  === {label} ===")
        for pc in per_city:
            print(f"    {pc['city_id']}: CPC_full={pc['CPC_full']:.4f}  CPC_nz={pc['CPC_nz']:.4f}  CPC_test={pc['CPC_test']:.4f}")
        print(f"  Avg: CPC_full={avg_mf['CPC']:.4f}  CPC_nz={avg_mnz['CPC']:.4f}  CPC_test={avg_mt['CPC']:.4f}  MAE={avg_mf['MAE']:.4f}")
        return avg_mf, avg_mnz, avg_mt, per_city

    mf_last, mnz_last, mt_last, pc_last = eval_all(f"Last weights (epoch {epoch})")
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    mf_best, mnz_best, mt_best, pc_best = eval_all("Best weights (best val_loss)")

    if mnz_last['CPC'] > mnz_best['CPC']:
        print(f"\n  ! Last weights better by CPC_nz ({mnz_last['CPC']:.4f} > {mnz_best['CPC']:.4f})")
        mf, mnz, mt, pc = mf_last, mnz_last, mt_last, pc_last
    else:
        mf, mnz, mt, pc = mf_best, mnz_best, mt_best, pc_best

    val_loss_best_state = best_state if best_state is not None else last_state
    if mf_last['CPC'] > mf_best['CPC']:
        cpc_best_state = last_state
        cpc_best_label = f"last weights (epoch {epoch})"
        print(f"\n  ! Last weights better by CPC_full ({mf_last['CPC']:.4f} > {mf_best['CPC']:.4f})")
    else:
        cpc_best_state = val_loss_best_state
        cpc_best_label = "best val_loss weights"

    # Save
    save_metrics_to_csv(run_id, run_name, config, mf, mnz, mt, n_params, epoch, status)
    save_model_weights(run_id, val_loss_best_state, config)
    print(f"  -> CPC_full-best checkpoint source: {cpc_best_label}")
    save_model_weights(run_id, cpc_best_state, config, weights_dir=WEIGHTS_CPC_BEST_DIR)

    return {
        'name': run_name, 'model': model, 'config': config, 'history': history,
        'metrics_full': mf, 'metrics_nonzero': mnz, 'metrics_test_pairs': mt,
        'per_city': pc, 'status': status,
        'loss_plot_path': str(saved_plot_path) if saved_plot_path is not None else None,
    }


# ─── High-level train functions ──────────────────────────────────────────────

def train_single_city(run_id, run_name, config, city_data=None, area_id=None, data_path=None):
    if city_data is None:
        kwargs = {}
        if area_id is not None:
            kwargs['area_id'] = area_id
        if data_path is not None:
            kwargs['data_path'] = data_path
        city_data = prepare_single_city_data(pe_type=config.pe_type, **kwargs)

    model = make_model(config, graph_data_ref=city_data['graph_data'])
    cid = city_data['city_id']
    return _train_loop(run_id, run_name, config, model, {cid: city_data}, is_multi=False)


def train_multi_city(run_id, run_name, config, city_data_dict=None,
                     train_city_ids=None, val_city_ids=None, test_city_ids=None,
                     city_ids=None, data_path=None):
    if city_data_dict is None:
        kwargs = {}
        if city_ids is not None:
            kwargs['city_ids'] = city_ids
        if data_path is not None:
            kwargs['data_path'] = data_path
        city_data_dict, train_city_ids, val_city_ids, test_city_ids = prepare_multi_city_data(
            pe_type=config.pe_type, **kwargs
        )
    input_dim = city_data_dict[list(city_data_dict.keys())[0]]['graph_data'].x.shape[1]
    model = make_model(config, input_dim=input_dim, edge_dim=1)
    return _train_loop(run_id, run_name, config, model, city_data_dict,
                       is_multi=True, train_city_ids=train_city_ids,
                       val_city_ids=val_city_ids, test_city_ids=test_city_ids)


