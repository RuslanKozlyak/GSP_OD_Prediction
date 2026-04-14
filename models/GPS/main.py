import time
import numpy as np
import torch
from dataclasses import replace

from .config import (
    TrainingConfig, WEIGHTS_DIR, WEIGHTS_CPC_BEST_DIR, device,
    ORIGIN_BATCH_SIZE, DEST_BATCH_SIZE, NAN_BATCH_THRESHOLD,
    METRICS_VAL_LOSS_CSV, METRICS_CPC_NZ_BEST_CSV,
    save_metrics_to_csv, save_model_weights,
)
from .model import make_model
from .loss import compute_loss_for_city
from .metrics import evaluate_full_matrix, predict_full_matrix, summarize_prediction_metrics
from .data_load import prepare_single_city_data, prepare_multi_city_data
from .gan import ODSequenceDiscriminator, gan_step_for_city
from models.shared.metrics import (
    average_listed_metrics,
    average_matrix_split_metrics,
    compute_metrics,
    format_train_val_cpc_metrics,
    masked_split_metrics,
    _SPLIT_METRIC_NAMES,
)
from models.shared.plotting import save_loss_plot


_NAN_TRAIN_VAL = {
    f'{m}_{split}_{var}': float('nan')
    for m in _SPLIT_METRIC_NAMES
    for split in ('train', 'val')
    for var in ('full', 'nz')
}


# ─── Unified training loop ───────────────────────────────────────────────────

def _train_loop(run_id, run_name, config, model, city_datas,
                is_multi=False, train_city_ids=None, val_city_ids=None,
                test_city_ids=None):
    """
    Unified training loop for single-city and multi-city modes.
    city_datas: dict {city_id: city_data}
    """
    print(f"\n{'='*70}\n  {run_name}\n  {config.describe()}\n{'='*70}")
    n_params_model = sum(p.numel() for p in model.parameters())
    n_params = n_params_model
    print(f"  Params: {n_params_model:,}")

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

    discriminator = None
    discriminator_optimizer = None
    if config.training_mode == 'gan':
        disc_input_dim = sample_cd['graph_data'].x.size(-1) + 1
        discriminator = ODSequenceDiscriminator(
            disc_input_dim,
            hidden_dim=config.gan_disc_hidden_dim,
            n_layers=config.gan_disc_layers,
            dropout=config.gan_disc_dropout,
        ).to(device)
        n_params_disc = sum(p.numel() for p in discriminator.parameters())
        n_params += n_params_disc
        print(
            f"  GAN: discriminator_params={n_params_disc:,} "
            f"pretrain={config.gan_pretrain_epochs} epochs "
            f"ncritic={config.gan_n_critic} adv_weight={config.adv_weight:g} "
            f"regularizer={config.gan_regularizer}"
        )
        if config.gan_regularizer == 'clip':
            print(f"       WGAN clip={config.gan_clip_value:g}")
        if config.gan_n_critic_after_epoch:
            print(
                f"       ncritic schedule: {config.gan_n_critic} through epoch "
                f"{config.gan_n_critic_after_epoch}, then {config.gan_n_critic_after}"
            )
        if config.gan_noise_dim:
            print(f"       generator_noise_dim={config.gan_noise_dim}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if discriminator is not None:
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(0.5, 0.9),
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)
    use_supervised_monitoring = discriminator is None or config.gan_use_supervised_monitoring
    # GAN monitoring is diagnostics only: it must not drive LR scheduling,
    # early stopping, or checkpoint selection.
    use_training_control = discriminator is None

    max_epochs = config.mc_epochs if is_multi else config.epochs
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cpc_full': [],
        'train_cpc_nz': [],
        'val_cpc_full': [],
        'val_cpc_nz': [],
        'gan_g_loss': [],
        'gan_d_loss': [],
        'gan_gp': [],
    }
    best_val_loss = float('inf')
    patience_count = 0
    best_state = None
    best_val_epoch = 0
    best_cpc_nz_val = -float('inf')
    best_cpc_nz_state = None
    best_cpc_nz_epoch = 0
    best_gan_g_loss = float('inf')
    best_gan_state = None
    best_gan_epoch = 0

    # Split cities into train / val
    split_scope = 'multi_city' if is_multi else 'single_city'

    def _make_split_view(cd, split_name):
        view = dict(cd)
        if split_name == 'val':
            view['od_matrix_train'] = cd.get('od_matrix_val', cd['od_matrix_np'] * cd['val_mask'])
            view['outflow_train'] = cd['outflow_val']
            view['inflow_train'] = cd['inflow_val']
            view['nonzero_dest_dict'] = cd.get('val_dest_dict', cd['nonzero_dest_dict'])
            view['active_fit_mask'] = cd.get('val_fit_mask', cd.get('val_mask'))
            view['active_origin_indices'] = cd.get(
                'val_origin_indices',
                np.where(np.asarray(view['active_fit_mask'], dtype=bool).any(1))[0],
            )
            return view
        view['active_fit_mask'] = cd.get('train_fit_mask', cd.get('train_mask'))
        view['active_origin_indices'] = cd.get(
            'active_origin_indices',
            np.where(np.asarray(view['active_fit_mask'], dtype=bool).any(1))[0],
        )
        return view

    def _average_metric_dicts(metric_dicts):
        if not metric_dicts:
            return {}
        keys = set().union(*(d.keys() for d in metric_dicts))
        averaged = {}
        for key in keys:
            vals = []
            for metrics in metric_dicts:
                value = metrics.get(key, float('nan'))
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(value):
                    vals.append(value)
            averaged[key] = float(np.mean(vals)) if vals else float('nan')
        return averaged

    if is_multi:
        assert train_city_ids is not None and val_city_ids is not None
        train_cds = {
            cid: _make_split_view(city_datas[cid], 'train')
            for cid in train_city_ids if cid in city_datas
        }
        val_cds = {
            cid: _make_split_view(city_datas[cid], 'val')
            for cid in val_city_ids if cid in city_datas
        }
        if test_city_ids is None:
            seen = set(train_city_ids) | set(val_city_ids)
            test_city_ids = [cid for cid in city_datas if cid not in seen]
    else:
        cid = list(city_datas.keys())[0]
        cd = city_datas[cid]
        train_cds = {cid: _make_split_view(cd, 'train')}
        val_cds = {cid: _make_split_view(cd, 'val')}

    def _train_val_metrics():
        if not is_multi:
            full_cd = list(city_datas.values())[0]
            pred = predict_full_matrix(model, full_cd, config, DEST_BATCH_SIZE)
            return masked_split_metrics(
                pred,
                full_cd['od_matrix_np'],
                full_cd['train_mask'],
                full_cd['val_mask'],
                train_full_mask=full_cd.get('train_full_mask'),
                val_full_mask=full_cd.get('val_full_mask'),
            )

        metrics = dict(_NAN_TRAIN_VAL)
        train_group_metrics = []
        for cid in train_cds:
            full_cd = city_datas[cid]
            pred = predict_full_matrix(model, full_cd, config, DEST_BATCH_SIZE)
            train_group_metrics.append(masked_split_metrics(
                pred,
                full_cd['od_matrix_np'],
                full_cd['train_mask'],
                full_cd['val_mask'],
                train_full_mask=full_cd.get('train_full_mask'),
                val_full_mask=full_cd.get('val_full_mask'),
            ))
        val_group_metrics = []
        for cid in val_cds:
            full_cd = city_datas[cid]
            pred = predict_full_matrix(model, full_cd, config, DEST_BATCH_SIZE)
            val_group_metrics.append(masked_split_metrics(
                pred,
                full_cd['od_matrix_np'],
                full_cd['train_mask'],
                full_cd['val_mask'],
                train_full_mask=full_cd.get('train_full_mask'),
                val_full_mask=full_cd.get('val_full_mask'),
            ))
        avg_train = _average_metric_dicts(train_group_metrics)
        avg_val = _average_metric_dicts(val_group_metrics)
        for key in metrics:
            if '_train_' in key:
                metrics[key] = avg_train.get(key, float('nan'))
            elif '_val_' in key:
                metrics[key] = avg_val.get(key, float('nan'))
        return metrics

    epoch = 0
    status = 'ok'
    loss_plot_path = WEIGHTS_DIR.parent / "loss_plots" / f"{run_id}_loss.png"
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        city_ids_shuffled = list(train_cds.keys())
        np.random.shuffle(city_ids_shuffled)
        epoch_losses = []
        epoch_gan_g_losses = []
        epoch_gan_d_losses = []
        epoch_gan_gps = []
        nan_count = 0
        total_batches = 0
        gan_nan_count = 0
        total_gan_steps = 0
        gan_only_epoch = (
            discriminator is not None
            and config.gan_only
            and epoch > config.gan_pretrain_epochs
        )

        for cid in city_ids_shuffled:
            cd = train_cds[cid]
            cc = replace(config, n_dest_sample=cd.get('city_n_dest', config.n_dest_sample)) if is_multi else config
            if not gan_only_epoch:
                origins = np.array(cd.get('active_origin_indices', list(cd['nonzero_dest_dict'].keys())))
                np.random.shuffle(origins)
                for bs in range(0, len(origins), ORIGIN_BATCH_SIZE):
                    batch = origins[bs:bs + ORIGIN_BATCH_SIZE].tolist()
                    optimizer.zero_grad()
                    # Encode inside the batch — each backward pass needs its
                    # own computation graph rooted at the encoder output.
                    loss = compute_loss_for_city(model, cd, cc, origin_batch_indices=batch)
                    total_batches += 1
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())

            if discriminator is not None and epoch > config.gan_pretrain_epochs:
                total_gan_steps += 1
                gan_stats = gan_step_for_city(
                    model,
                    discriminator,
                    cd,
                    cc,
                    optimizer,
                    discriminator_optimizer,
                    epoch=epoch,
                )
                gan_invalid = False
                for key, target in (
                    ('gan_g_loss', epoch_gan_g_losses),
                    ('gan_d_loss', epoch_gan_d_losses),
                    ('gan_gp', epoch_gan_gps),
                ):
                    value = gan_stats.get(key, float('nan'))
                    if np.isnan(value):
                        if key in ('gan_g_loss', 'gan_d_loss'):
                            gan_invalid = True
                    else:
                        target.append(value)
                if gan_invalid:
                    gan_nan_count += 1

        # Divergence check
        total_steps = total_batches + total_gan_steps
        total_nan = nan_count + gan_nan_count
        if total_steps > 0 and total_nan / total_steps > NAN_BATCH_THRESHOLD:
            print(
                f"  NaN divergence @ epoch {epoch} "
                f"(sup={nan_count}/{total_batches}, gan={gan_nan_count}/{total_gan_steps})"
            )
            status = 'nan_diverged'
            break

        train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        gan_g_loss = np.mean(epoch_gan_g_losses) if epoch_gan_g_losses else float('nan')
        gan_d_loss = np.mean(epoch_gan_d_losses) if epoch_gan_d_losses else float('nan')
        gan_gp = np.mean(epoch_gan_gps) if epoch_gan_gps else float('nan')

        # Optional GAN monitoring only computes validation diagnostics.
        if use_supervised_monitoring:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vcid, vcd in val_cds.items():
                    vcfg = replace(config, n_dest_sample=vcd.get('city_n_dest', config.n_dest_sample)) if is_multi else config
                    vl = compute_loss_for_city(model, vcd, vcfg)
                    if not (torch.isnan(vl) or torch.isinf(vl)):
                        val_losses.append(vl.item())
                train_val_cpc = _train_val_metrics()
            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        else:
            train_val_cpc = dict(_NAN_TRAIN_VAL)
            avg_val_loss = float('nan')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_cpc_full'].append(train_val_cpc['CPC_train_full'])
        history['train_cpc_nz'].append(train_val_cpc['CPC_train_nz'])
        history['val_cpc_full'].append(train_val_cpc['CPC_val_full'])
        history['val_cpc_nz'].append(train_val_cpc['CPC_val_nz'])
        history['gan_g_loss'].append(gan_g_loss)
        history['gan_d_loss'].append(gan_d_loss)
        history['gan_gp'].append(gan_gp)

        if discriminator is not None and not np.isnan(gan_g_loss) and gan_g_loss < best_gan_g_loss:
            best_gan_g_loss = gan_g_loss
            best_gan_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_gan_epoch = epoch
            flag = ' *'
        elif discriminator is not None:
            flag = ''

        if use_training_control and not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)

        if use_training_control and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch
            patience_count = 0
            flag = ' *'
        elif use_training_control:
            patience_count += 1
            flag = ''
        else:
            flag = ''

        if use_training_control:
            cpc_nz_val = train_val_cpc['CPC_val_nz']
            if not np.isnan(cpc_nz_val) and cpc_nz_val > best_cpc_nz_val:
                best_cpc_nz_val = cpc_nz_val
                best_cpc_nz_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_cpc_nz_epoch = epoch

        if epoch % 5 == 0 or epoch == 1:
            nan_parts = []
            if nan_count > 0:
                nan_parts.append(f"sup={nan_count}")
            if gan_nan_count > 0:
                nan_parts.append(f"gan={gan_nan_count}")
            nan_str = f" NaN:{','.join(nan_parts)}" if nan_parts else ""
            train_text = "gan_only" if gan_only_epoch else f"{train_loss:.4f}"
            gan_str = ""
            if discriminator is not None:
                if epoch <= config.gan_pretrain_epochs:
                    gan_str = " gan=pretrain"
                else:
                    d_text = "-" if np.isnan(gan_d_loss) else f"{gan_d_loss:.4f}"
                    g_text = "-" if np.isnan(gan_g_loss) else f"{gan_g_loss:.4f}"
                    gan_str = f" gan_g={g_text} gan_d={d_text}"
            monitor_text = (
                f"val={avg_val_loss:.4f}  {format_train_val_cpc_metrics(train_val_cpc)}"
                if use_supervised_monitoring else
                "val=off  CPC_train/val=off"
            )
            print(f"  {epoch:3d}/{max_epochs}  train={train_text}  {monitor_text}  "
                  f"{time.time()-t0:.1f}s{flag}{nan_str}{gan_str}")
        if use_training_control and patience_count >= config.patience:
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
        dummy = {
            'CPC_full': 0.0, 'MAE_full': float('inf'), 'RMSE_full': float('inf'),
            'CPC_nz': 0.0, 'MAE_nz': float('inf'), 'RMSE_nz': float('inf'),
        }
        for metrics_csv, run_suffix, selection in (
            (
                METRICS_VAL_LOSS_CSV,
                'val_loss',
                ('val_loss_best' if use_training_control else 'gan_g_loss_best'),
            ),
            (
                METRICS_CPC_NZ_BEST_CSV,
                'cpc_nz',
                ('cpc_nz_best' if use_training_control else 'gan_g_loss_best'),
            ),
        ):
            save_metrics_to_csv(
                run_id, run_name, config, dummy,
                n_params, epoch, status,
                metrics_csv=metrics_csv,
                run_suffix=run_suffix,
                checkpoint_selection=selection,
                split_scope=split_scope,
            )
        return {
            'name': run_name, 'model': model, 'config': config, 'history': history,
            'discriminator': discriminator,
            'metrics': dummy,
            'pred_matrix': None, 'status': status,
            'loss_plot_path': str(saved_plot_path) if saved_plot_path is not None else None,
        }

    # Evaluate selected checkpoints
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
        """Evaluate all cities and return averaged canonical metrics."""
        all_combined = []
        per_city = []
        test_eval_ids = set(test_eval_cities)

        def fmt(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "-"
            return "-" if np.isnan(v) else f"{v:.4f}"

        for ecid, ecd in full_eval_cities.items():
            is_test = (ecid in test_eval_ids) if is_multi else True
            pred, combined = evaluate_full_matrix(
                model, ecd, config, dest_batch_size=DEST_BATCH_SIZE,
                is_test_city=is_test,
            )
            if is_multi and not is_test:
                combined = {
                    key: (float('nan') if '_test_' in key else value)
                    for key, value in combined.items()
                }
            all_combined.append(combined)
            per_city.append({
                'city_id': ecid,
                'CPC_full': combined['CPC_full'],
                'CPC_nz': combined['CPC_nz'],
                'MAE_full': combined['MAE_full'],
                'RMSE_full': combined['RMSE_full'],
                'CPC_test_full': combined.get('CPC_test_full', float('nan')),
                'CPC_test_nz': combined.get('CPC_test_nz', float('nan')),
            })

        avg = average_listed_metrics(all_combined)
        print(f"\n  === {label} ===")
        for pc in per_city:
            print(
                f"    {pc['city_id']}: CPC_full={fmt(pc['CPC_full'])}  "
                f"CPC_nz={fmt(pc['CPC_nz'])}  "
                f"CPC_test_full={fmt(pc['CPC_test_full'])}  "
                f"CPC_test_nz={fmt(pc['CPC_test_nz'])}"
            )
        print(
            f"  Avg: CPC_full={fmt(avg.get('CPC_full'))}  "
            f"CPC_nz={fmt(avg.get('CPC_nz'))}  "
            f"CPC_test_full={fmt(avg.get('CPC_test_full'))}  "
            f"CPC_test_nz={fmt(avg.get('CPC_test_nz'))}  "
            f"MAE={fmt(avg.get('MAE_full'))}"
        )
        return avg, per_city

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    val_loss_best_state = best_state if best_state is not None else last_state
    cpc_nz_best_state = best_cpc_nz_state if best_cpc_nz_state is not None else val_loss_best_state
    gan_best_state = best_gan_state if best_gan_state is not None else last_state

    if use_training_control:
        model.load_state_dict(val_loss_best_state)
        tv_val = _train_val_metrics()
        avg_val, pc_val = eval_all(
            f"Best weights (val_loss @ epoch {best_val_epoch or epoch})"
        )
        avg_val.update(tv_val)
        print(f"  Train/Val: {format_train_val_cpc_metrics(avg_val)}")

        model.load_state_dict(cpc_nz_best_state)
        tv_cpc = _train_val_metrics()
        avg_cpc, pc_cpc = eval_all(
            f"Best weights (CPC_val_nz @ epoch {best_cpc_nz_epoch or best_val_epoch or epoch})"
        )
        avg_cpc.update(tv_cpc)
        print(f"  Train/Val: {format_train_val_cpc_metrics(avg_cpc)}")
    else:
        model.load_state_dict(gan_best_state)
        avg_val, pc_val = eval_all(
            f"Best weights (gan_g_loss @ epoch {best_gan_epoch or epoch})"
        )
        avg_val.update(_NAN_TRAIN_VAL)
        avg_cpc, pc_cpc = avg_val, pc_val

    pure_gan_selected_epoch = best_gan_epoch or epoch
    pure_gan_selection_value = best_gan_g_loss if np.isfinite(best_gan_g_loss) else None

    # Save
    save_metrics_to_csv(
        run_id, run_name, config, avg_val,
        n_params, epoch, status,
        metrics_csv=METRICS_VAL_LOSS_CSV,
        run_suffix='val_loss',
        checkpoint_selection='val_loss_best' if use_training_control else 'gan_g_loss_best',
        selected_epoch=(
            (best_val_epoch or epoch)
            if use_training_control else pure_gan_selected_epoch
        ),
        selection_metric='val_loss' if use_training_control else 'gan_g_loss',
        selection_metric_value=best_val_loss if use_training_control else pure_gan_selection_value,
        split_scope=split_scope,
    )
    save_metrics_to_csv(
        run_id, run_name, config, avg_cpc,
        n_params, epoch, status,
        metrics_csv=METRICS_CPC_NZ_BEST_CSV,
        run_suffix='cpc_nz',
        checkpoint_selection='cpc_nz_best' if use_training_control else 'gan_g_loss_best',
        selected_epoch=(
            (best_cpc_nz_epoch or best_val_epoch or epoch)
            if use_training_control else pure_gan_selected_epoch
        ),
        selection_metric='CPC_val_nz' if use_training_control else 'gan_g_loss',
        selection_metric_value=best_cpc_nz_val if use_training_control else pure_gan_selection_value,
        split_scope=split_scope,
    )
    export_state = (
        val_loss_best_state
        if use_training_control else
        (gan_best_state if discriminator is not None else val_loss_best_state)
    )
    save_model_weights(run_id, export_state, config)
    if use_training_control:
        print(
            "  -> CPC_nz-best checkpoint source: "
            f"epoch {best_cpc_nz_epoch or best_val_epoch or epoch}"
        )
    else:
        print(f"  -> Pure GAN checkpoint source: gan_g_loss-best epoch {pure_gan_selected_epoch}")
    cpc_export_state = cpc_nz_best_state if use_training_control else gan_best_state
    save_model_weights(run_id, cpc_export_state, config, weights_dir=WEIGHTS_CPC_BEST_DIR)

    return {
        'name': run_name, 'model': model, 'config': config, 'history': history,
        'discriminator': discriminator,
        'metrics': avg_cpc,
        'metrics_val_loss': avg_val,
        'per_city': pc_cpc, 'per_city_val_loss': pc_val, 'status': status,
        'loss_plot_path': str(saved_plot_path) if saved_plot_path is not None else None,
    }


# ─── High-level train functions ──────────────────────────────────────────────

def train_single_city(run_id, run_name, config, city_data=None, area_id=None, data_path=None):
    if city_data is None or city_data.get('pair_split_mode') != config.pair_split_mode:
        kwargs = {}
        effective_area_id = area_id if area_id is not None else (
            city_data.get('city_id') if city_data is not None else None
        )
        if effective_area_id is not None:
            kwargs['area_id'] = effective_area_id
        if data_path is not None:
            kwargs['data_path'] = data_path
        city_data = prepare_single_city_data(
            pe_type=config.pe_type,
            pair_split_mode=config.pair_split_mode,
            **kwargs,
        )

    model = make_model(config, graph_data_ref=city_data['graph_data'])
    cid = city_data['city_id']
    return _train_loop(run_id, run_name, config, model, {cid: city_data}, is_multi=False)


def train_multi_city(run_id, run_name, config, city_data_dict=None,
                     train_city_ids=None, val_city_ids=None, test_city_ids=None,
                     city_ids=None, data_path=None):
    needs_reload = city_data_dict is None or any(
        cd.get('pair_split_mode') != config.pair_split_mode
        or cd.get('feature_scaling_scope') != 'global_train_cities'
        for cd in (city_data_dict or {}).values()
    )
    if needs_reload:
        kwargs = {}
        effective_city_ids = city_ids if city_ids is not None else (
            list(city_data_dict.keys()) if city_data_dict is not None else None
        )
        if effective_city_ids is not None:
            kwargs['city_ids'] = effective_city_ids
        if data_path is not None:
            kwargs['data_path'] = data_path
        city_data_dict, train_city_ids, val_city_ids, test_city_ids = prepare_multi_city_data(
            pe_type=config.pe_type,
            pair_split_mode=config.pair_split_mode,
            **kwargs,
        )
    input_dim = city_data_dict[list(city_data_dict.keys())[0]]['graph_data'].x.shape[1]
    model = make_model(config, input_dim=input_dim, edge_dim=1)
    return _train_loop(run_id, run_name, config, model, city_data_dict,
                       is_multi=True, train_city_ids=train_city_ids,
                       val_city_ids=val_city_ids, test_city_ids=test_city_ids)
