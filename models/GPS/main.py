import time
import numpy as np
import torch
from dataclasses import replace

from .config import (
    TrainingConfig, device,
    ORIGIN_BATCH_SIZE, DEST_BATCH_SIZE, NAN_BATCH_THRESHOLD, MC_EPOCHS,
    save_metrics_to_csv, save_model_weights,
)
from .model import make_model
from .loss import compute_loss_for_city
from .metrics import compute_metrics, evaluate_full_matrix
from .data_load import prepare_single_city_data, prepare_multi_city_data


# ─── Unified training loop ───────────────────────────────────────────────────

def _train_loop(run_id, run_name, config, model, city_datas,
                is_multi=False, train_city_ids=None, val_city_ids=None):
    """
    Unified training loop for single-city and multi-city modes.
    city_datas: dict {city_id: city_data}
    """
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

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

    if status == 'nan_diverged':
        dummy = {'CPC': 0.0, 'MAE': float('inf'), 'RMSE': float('inf')}
        save_metrics_to_csv(run_id, run_name, config, dummy, dummy, dummy, n_params, epoch, status)
        return {
            'name': run_name, 'model': model, 'config': config, 'history': history,
            'metrics_full': dummy, 'metrics_nonzero': dummy, 'metrics_test_pairs': dummy,
            'pred_matrix': None, 'status': status,
        }

    # Evaluate on last and best weights
    eval_cities = city_datas if is_multi else {list(city_datas.keys())[0]: list(city_datas.values())[0]}

    def eval_all(label):
        all_mf = []
        all_mnz = []
        all_mt = []
        per_city = []
        for ecid, ecd in eval_cities.items():
            pred, mf, mnz = evaluate_full_matrix(model, ecd, config, dest_batch_size=DEST_BATCH_SIZE)
            mt = compute_metrics(pred[ecd['test_mask']], ecd['od_matrix_np'][ecd['test_mask']].astype(float))
            all_mf.append(mf)
            all_mnz.append(mnz)
            all_mt.append(mt)
            per_city.append({
                'city_id': ecid, 'CPC_full': mf['CPC'], 'CPC_nz': mnz['CPC'],
                'MAE': mf['MAE'], 'RMSE': mf['RMSE'], 'CPC_test': mt['CPC'],
            })
        avg_mf = {k: np.mean([m[k] for m in all_mf]) for k in all_mf[0]}
        avg_mnz = {k: np.mean([m[k] for m in all_mnz]) for k in all_mnz[0]}
        avg_mt = {k: np.mean([m[k] for m in all_mt]) for k in all_mt[0]}
        print(f"\n  === {label} ===")
        for pc in per_city:
            print(f"    {pc['city_id']}: CPC_full={pc['CPC_full']:.4f}  CPC_nz={pc['CPC_nz']:.4f}  CPC_test={pc['CPC_test']:.4f}")
        print(f"  Avg: CPC_full={avg_mf['CPC']:.4f}  CPC_nz={avg_mnz['CPC']:.4f}  CPC_test={avg_mt['CPC']:.4f}  MAE={avg_mf['MAE']:.4f}")
        return avg_mf, avg_mnz, avg_mt, per_city

    mf_last, mnz_last, mt_last, pc_last = eval_all(f"Last weights (epoch {epoch})")

    if best_state:
        model.load_state_dict(best_state)
    mf_best, mnz_best, mt_best, pc_best = eval_all("Best weights (best val_loss)")

    if mnz_last['CPC'] > mnz_best['CPC']:
        print(f"\n  ! Last weights better by CPC_nz ({mnz_last['CPC']:.4f} > {mnz_best['CPC']:.4f})")
        mf, mnz, mt, pc = mf_last, mnz_last, mt_last, pc_last
    else:
        mf, mnz, mt, pc = mf_best, mnz_best, mt_best, pc_best

    # Save
    save_metrics_to_csv(run_id, run_name, config, mf, mnz, mt, n_params, epoch, status)
    save_model_weights(run_id, model)

    return {
        'name': run_name, 'model': model, 'config': config, 'history': history,
        'metrics_full': mf, 'metrics_nonzero': mnz, 'metrics_test_pairs': mt,
        'per_city': pc, 'status': status,
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
                     train_city_ids=None, val_city_ids=None, city_ids=None, data_path=None):
    if city_data_dict is None:
        kwargs = {}
        if city_ids is not None:
            kwargs['city_ids'] = city_ids
        if data_path is not None:
            kwargs['data_path'] = data_path
        city_data_dict, train_city_ids, val_city_ids, _ = prepare_multi_city_data(
            pe_type=config.pe_type, **kwargs
        )
    input_dim = city_data_dict[list(city_data_dict.keys())[0]]['graph_data'].x.shape[1]
    model = make_model(config, input_dim=input_dim, edge_dim=1)
    return _train_loop(run_id, run_name, config, model, city_data_dict,
                       is_multi=True, train_city_ids=train_city_ids, val_city_ids=val_city_ids)


# ─── Experiment definitions ──────────────────────────────────────────────────

SINGLE_CITY_RUNS = [
    ('B1', 'B1: GPS+Bilinear+Huber (raw)', TrainingConfig(decoder_type='bilinear', loss_type='huber', prediction_mode='raw')),
    ('B2', 'B2: GPS+TransFlower+Huber (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw')),
    ('B3', 'B3: GPS+Bilinear+Multitask (raw)', TrainingConfig(decoder_type='bilinear', loss_type='multitask', prediction_mode='raw')),
    ('B4', 'B4: GPS+TransFlower+Multitask (raw)', TrainingConfig(decoder_type='transflower', loss_type='multitask', prediction_mode='raw')),
    ('B6', 'B6: GPS+Bilinear+CE (normalized)', TrainingConfig(decoder_type='bilinear', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False)),
    ('B7', 'B7: GPS+TransFlower+CE (normalized)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False)),
    ('B2n', 'B2n: GPS+TransFlower+Huber (normalized)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='normalized')),
    # Huber ablations
    ('B2+spe', 'B2+SPE: GPS+TF+Huber+SPE (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='spe')),
    ('B2+rrwp', 'B2+RRWP: GPS+TF+Huber+RRWP (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='rrwp')),
    ('B2+gnorm', 'B2+GraphNorm: GPS+TF+Huber+GraphNorm (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', gps_norm_type='graph_norm')),
    ('B2+zinb', 'B2+ZINB: GPS+TF+ZINB (raw)', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='raw', include_zero_pairs=True, zero_pair_ratio=0.5)),
    ('B2+log', 'B2+Log: GPS+TF+Huber+Log (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_log_transform=True)),
    # CE ablations
    ('B7+spe', 'B7+SPE: GPS+TF+CE+SPE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, pe_type='spe')),
    ('B7+rrwp', 'B7+RRWP: GPS+TF+CE+RRWP (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, pe_type='rrwp')),
    ('B7+gnorm', 'B7+GraphNorm: GPS+TF+CE+GraphNorm (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, gps_norm_type='graph_norm')),
    # CE new single modifications
    ('B7+log', 'B7+Log: GPS+TF+CE+Log (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, use_log_transform=True)),
    ('B7+zinb', 'B7+ZINB: GPS+TF+ZINB (norm)', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='normalized', use_dest_sampling=False, include_zero_pairs=True, zero_pair_ratio=0.5)),
    # CE sampling
    ('B7+samp', 'B7+samp128: GPS+TF+CE+samp128', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=False)),
    ('B7+samp256', 'B7+samp256: GPS+TF+CE+samp256', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=256, include_zero_pairs=False)),
    ('B7+sz30', 'B7+sz30: GPS+TF+CE+samp+zeros30%', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.3)),
    ('B7+sz50', 'B7+sz50: GPS+TF+CE+samp+zeros50%', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.5)),
    # Combinations
    ('B2+combo1', 'B2+SPE+GraphNorm: GPS+TF+Huber+SPE+GN (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='spe', gps_norm_type='graph_norm')),
    ('B2+combo2', 'B2+SPE+ZINB: GPS+TF+ZINB+SPE (raw)', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='raw', pe_type='spe', include_zero_pairs=True, zero_pair_ratio=0.5)),
    ('B7+combo1', 'B7+SPE+GraphNorm: GPS+TF+CE+SPE+GN (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, pe_type='spe', gps_norm_type='graph_norm')),
    # Tier 2 combinations
    ('B2+log+spe', 'B2+Log+SPE: GPS+TF+Huber+Log+SPE (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_log_transform=True, pe_type='spe')),
    ('B7+sz30+log', 'B7+sz30+Log: GPS+TF+CE+sz30+Log', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.3, use_log_transform=True)),
    ('B7+sz50+log', 'B7+sz50+Log: GPS+TF+CE+sz50+Log', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.5, use_log_transform=True)),
    # Tier 3 exploratory
    ('B7+sz30+spe', 'B7+sz30+SPE: GPS+TF+CE+sz30+SPE', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.3, pe_type='spe')),
    # TransFlower (MLP encoder, no graph)
    ('TF1', 'TF1: TransFlower+CE (norm)', TrainingConfig(encoder_type='mlp', decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False)),
    ('TF1+rle', 'TF1+RLE: TransFlower+CE+RLE (norm)', TrainingConfig(encoder_type='mlp', decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, use_rle=True)),
    # GPS + RLE
    ('B7+rle', 'B7+RLE: GPS+TF+CE+RLE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, use_rle=True)),
    # LaPE
    ('B7+lape', 'B7+LaPE: GPS+TF+CE+LaPE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, pe_type='lape')),
    # Focal loss
    ('B7+focal', 'B7+Focal: GPS+TF+Focal (norm)', TrainingConfig(decoder_type='transflower', loss_type='focal', prediction_mode='normalized', use_dest_sampling=False)),
    ('TF1+focal', 'TF1+Focal: TransFlower+Focal (norm)', TrainingConfig(encoder_type='mlp', decoder_type='transflower', loss_type='focal', prediction_mode='normalized', use_dest_sampling=False)),
]

MULTI_CITY_RUNS = [
    ('C1', 'C1: MC GPS+TF+Huber (raw)', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', mc_epochs=MC_EPOCHS)),
    ('C2', 'C2: MC GPS+TF+CE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, mc_epochs=MC_EPOCHS)),
    ('C3', 'C3: MC GPS+TF+Multitask (raw)', TrainingConfig(decoder_type='transflower', loss_type='multitask', prediction_mode='raw', mc_epochs=MC_EPOCHS)),
    # Huber ablations
    ('C1+spe', 'C1+SPE: MC GPS+TF+Huber+SPE', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='spe', mc_epochs=MC_EPOCHS)),
    ('C1+rrwp', 'C1+RRWP: MC GPS+TF+Huber+RRWP', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='rrwp', mc_epochs=MC_EPOCHS)),
    ('C1+gnorm', 'C1+GraphNorm: MC GPS+TF+Huber+GN', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    ('C1+zinb', 'C1+ZINB: MC GPS+TF+ZINB', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='raw', include_zero_pairs=True, zero_pair_ratio=0.5, mc_epochs=MC_EPOCHS)),
    ('C1+log', 'C1+Log: MC GPS+TF+Huber+Log', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_log_transform=True, mc_epochs=MC_EPOCHS)),
    # CE ablations
    ('C2+spe', 'C2+SPE: MC GPS+TF+CE+SPE', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, pe_type='spe', mc_epochs=MC_EPOCHS)),
    ('C2+rrwp', 'C2+RRWP: MC GPS+TF+CE+RRWP', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, pe_type='rrwp', mc_epochs=MC_EPOCHS)),
    ('C2+gnorm', 'C2+GraphNorm: MC GPS+TF+CE+GN', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    # CE new single modifications
    ('C2+log', 'C2+Log: MC GPS+TF+CE+Log', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, use_log_transform=True, mc_epochs=MC_EPOCHS)),
    # Zeros-sampling (Huber)
    ('C1+sz30', 'C1+sz30: MC GPS+TF+Huber+sz30', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.3, mc_epochs=MC_EPOCHS)),
    ('C1+sz50', 'C1+sz50: MC GPS+TF+Huber+sz50', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.5, mc_epochs=MC_EPOCHS)),
    # Zeros-sampling (CE)
    ('C2+sz30', 'C2+sz30: MC GPS+TF+CE+sz30', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.3, mc_epochs=MC_EPOCHS)),
    ('C2+sz50', 'C2+sz50: MC GPS+TF+CE+sz50', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, n_dest_sample=128, include_zero_pairs=True, zero_pair_ratio=0.5, mc_epochs=MC_EPOCHS)),
    # Combinations
    ('C1+combo1', 'C1+SPE+GN: MC GPS+TF+Huber+SPE+GN', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', pe_type='spe', gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    ('C1+combo2', 'C1+SPE+ZINB: MC GPS+TF+ZINB+SPE', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='raw', pe_type='spe', include_zero_pairs=True, zero_pair_ratio=0.5, mc_epochs=MC_EPOCHS)),
    ('C2+combo1', 'C2+SPE+GN: MC GPS+TF+CE+SPE+GN', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, pe_type='spe', gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    # Tier 2 combinations
    ('C1+log+gnorm', 'C1+Log+GN: MC GPS+TF+Huber+Log+GN', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_log_transform=True, gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    ('C2+spe+zinb', 'C2+SPE+ZINB: MC GPS+TF+ZINB+SPE', TrainingConfig(decoder_type='transflower', loss_type='zinb', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=True, zero_pair_ratio=0.5, pe_type='spe', mc_epochs=MC_EPOCHS)),
    # Tier 3 exploratory
    ('C1+log+spe+gnorm', 'C1+Log+SPE+GN: MC GPS+TF+Huber+Log+SPE+GN', TrainingConfig(decoder_type='transflower', loss_type='huber', prediction_mode='raw', use_log_transform=True, pe_type='spe', gps_norm_type='graph_norm', mc_epochs=MC_EPOCHS)),
    # TransFlower (MLP encoder, no graph)
    ('TC1', 'TC1: MC TransFlower+CE (norm)', TrainingConfig(encoder_type='mlp', decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, mc_epochs=MC_EPOCHS)),
    ('TC1+rle', 'TC1+RLE: MC TransFlower+CE+RLE (norm)', TrainingConfig(encoder_type='mlp', decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=False, use_rle=True, mc_epochs=MC_EPOCHS)),
    # GPS + RLE
    ('C2+rle', 'C2+RLE: MC GPS+TF+CE+RLE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, use_rle=True, mc_epochs=MC_EPOCHS)),
    # LaPE
    ('C2+lape', 'C2+LaPE: MC GPS+TF+CE+LaPE (norm)', TrainingConfig(decoder_type='transflower', loss_type='ce', prediction_mode='normalized', use_dest_sampling=True, include_zero_pairs=False, pe_type='lape', mc_epochs=MC_EPOCHS)),
    # Focal loss
    ('C2+focal', 'C2+Focal: MC GPS+TF+Focal (norm)', TrainingConfig(decoder_type='transflower', loss_type='focal', prediction_mode='normalized', use_dest_sampling=False, mc_epochs=MC_EPOCHS)),
]


# ─── Run experiment (callable from benchmark) ────────────────────────────────

def run_experiment(run_id, run_name, config, mode='single', city_data=None,
                   city_data_dict=None, train_city_ids=None, val_city_ids=None,
                   area_id=None, city_ids=None, data_path=None):
    """
    Unified entry point. Returns result dict with metrics_full, metrics_nonzero, model, etc.
    mode: 'single' or 'multi'
    """
    if mode == 'single':
        return train_single_city(run_id, run_name, config, city_data=city_data,
                                 area_id=area_id, data_path=data_path)
    elif mode == 'multi':
        return train_multi_city(run_id, run_name, config, city_data_dict=city_data_dict,
                                train_city_ids=train_city_ids, val_city_ids=val_city_ids,
                                city_ids=city_ids, data_path=data_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPS OD Model Training")
    parser.add_argument('--mode', choices=['single', 'multi', 'all'], default='single',
                        help='Training mode: single-city, multi-city, or all experiments')
    parser.add_argument('--run_ids', nargs='+', default=None,
                        help='Specific run IDs to execute (e.g. B2 B7 C1). Default: all.')
    parser.add_argument('--data_path', default=None, help='Override data path')
    args = parser.parse_args()

    if args.mode in ('single', 'all'):
        run_ids = args.run_ids
        results = {}
        for rid, rname, rcfg in SINGLE_CITY_RUNS:
            if run_ids and rid not in run_ids:
                continue
            results[rid] = train_single_city(rid, rname, rcfg, data_path=args.data_path)
        print(f"\nCompleted {len(results)} single-city experiments")

    if args.mode in ('multi', 'all'):
        run_ids = args.run_ids
        results = {}
        for rid, rname, rcfg in MULTI_CITY_RUNS:
            if run_ids and rid not in run_ids:
                continue
            results[rid] = train_multi_city(rid, rname, rcfg, data_path=args.data_path)
        print(f"\nCompleted {len(results)} multi-city experiments")
