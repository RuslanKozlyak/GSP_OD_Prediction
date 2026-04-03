import gc
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from models.GPS.metrics import average_listed_metrics, cal_od_metrics

from .config import (
    DATA_PATH,
    FLAT_BATCH_SIZE,
    FLAT_SGD_EPOCHS,
    PROJECT_ROOT,
    SEED,
    cleanup_gpu,
    device,
)
from .data_utils import (
    build_dgl_graph,
    construct_flat_features,
    count_chunks,
    get_graph_scalers,
    iter_flat_chunks,
    iter_graph_areas,
    load_graph_data,
)



def load_model_main(model_name):
    model_dir = PROJECT_ROOT / "models" / model_name
    spec = importlib.util.spec_from_file_location(f"{model_name}_main", model_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def run_flat_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    feature_mode = "gravity" if model_name in ("GM_E", "GM_P") else "full"
    model = None

    if model_name == "RF":
        from sklearn.ensemble import RandomForestRegressor

        n_chunks = count_chunks(train_areas)
        trees_per_chunk = max(2, 20 // n_chunks)
        total_trees = 0
        model = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        )
        for idx, (x_chunk, y_chunk) in enumerate(
            tqdm(iter_flat_chunks(train_areas, data_path, feature_mode), total=n_chunks, desc="RF chunks")
        ):
            total_trees += trees_per_chunk
            model.n_estimators = total_trees
            model.fit(x_chunk, y_chunk)
            tqdm.write(f"    Chunk {idx + 1}/{n_chunks}: {len(x_chunk)} samples, {total_trees} trees")
            del x_chunk, y_chunk
            gc.collect()
        print(f"  RF trained: {total_trees} trees total")

    elif model_name == "SVR":
        from sklearn.linear_model import SGDRegressor

        n_chunks = count_chunks(train_areas)
        scaler = StandardScaler()
        total_samples = 0
        for x_chunk, _ in tqdm(iter_flat_chunks(train_areas, data_path, feature_mode), total=n_chunks, desc="SVR scaler"):
            scaler.partial_fit(x_chunk)
            total_samples += len(x_chunk)
            del x_chunk
            gc.collect()
        print(f"  Scaler fitted on {total_samples} total samples")

        model = SGDRegressor(
            loss="epsilon_insensitive",
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            random_state=SEED,
            warm_start=False,
        )
        for _ in tqdm(range(FLAT_SGD_EPOCHS), desc="SVR epochs"):
            for x_chunk, y_chunk in iter_flat_chunks(train_areas, data_path, feature_mode):
                model.partial_fit(scaler.transform(x_chunk), y_chunk)
                del x_chunk, y_chunk
                gc.collect()

        sgd_model, sgd_scaler = model, scaler

        class SVRWrapper:
            def predict(self, x):
                return sgd_model.predict(sgd_scaler.transform(x))

        model = SVRWrapper()

    elif model_name == "GBRT":
        from sklearn.ensemble import HistGradientBoostingRegressor

        x_parts, y_parts = [], []
        for x_chunk, y_chunk in tqdm(
            iter_flat_chunks(train_areas, data_path, feature_mode),
            total=count_chunks(train_areas),
            desc="GBRT load",
        ):
            x_parts.append(x_chunk)
            y_parts.append(y_chunk)
        x_train = np.concatenate(x_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)
        del x_parts, y_parts
        gc.collect()

        print(f"  GBRT training on {len(x_train)} samples ({x_train.nbytes / 1e6:.0f} MB)")
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=None,
            min_samples_leaf=2,
            random_state=SEED,
        )
        model.fit(x_train, y_train)
        del x_train, y_train
        gc.collect()

    elif model_name in ("DGM", "GM_E", "GM_P"):
        xs_tr, ys_tr = construct_flat_features(train_areas, data_path, feature_mode)
        x_train = np.concatenate(xs_tr, axis=0)
        y_train = np.concatenate(ys_tr, axis=0)
        del xs_tr, ys_tr
        gc.collect()

        xs_valid, ys_valid = construct_flat_features(valid_areas, data_path, feature_mode)
        model = load_model_main(model_name).train(
            x_train,
            y_train,
            xs_valid,
            ys_valid,
            device=device,
            batch_size=FLAT_BATCH_SIZE,
            max_epochs=10000,
            patience=100,
        )
        del x_train, y_train, xs_valid, ys_valid
        gc.collect()
    else:
        raise ValueError(f"Unsupported flat model: {model_name}")

    xs_test, ys_test = construct_flat_features(test_areas, data_path, feature_mode)
    metrics_all = []
    for x_one, y_one in zip(xs_test, ys_test):
        n_nodes = int(np.sqrt(y_one.shape[0]))
        y_hat = model.predict(x_one) if hasattr(model, "predict") else model(x_one)
        y_hat = y_hat.reshape(n_nodes, n_nodes)
        y_true = y_one.reshape(n_nodes, n_nodes)
        y_hat[y_hat < 0] = 0
        metrics_all.append(cal_od_metrics(y_hat, y_true))

    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    del model
    gc.collect()
    cleanup_gpu()
    return metrics_all



def run_graph_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    nf_valid, _, dis_valid, od_valid = load_graph_data(valid_areas, data_path)
    nf_test, adj_test, dis_test, od_test = load_graph_data(test_areas, data_path)
    nfeat_scaler, dis_scaler, od_scaler = get_graph_scalers(nf_valid, dis_valid, od_valid)
    metrics_all = []

    try:
        if model_name == "GMEL":
            gmel, gbrt, nfeat_scaler, dis_scaler = load_model_main("GMEL").train(
                train_areas,
                valid_areas,
                str(data_path),
                device=device,
                nfeat_scaler=nfeat_scaler,
                dis_scaler=dis_scaler,
                od_scaler=od_scaler,
                max_epochs=1000,
                patience=10,
            )
            gmel.eval()
            for nf, adj, dis, od in tqdm(zip(nf_test, adj_test, dis_test, od_test), total=len(nf_test), desc="GMEL eval"):
                nf_t = torch.FloatTensor(nfeat_scaler.transform(nf)).to(device)
                graph = build_dgl_graph(adj, device)
                with torch.no_grad():
                    _, _, _, h_in, h_out = gmel(graph, nf_t)
                    h = np.concatenate([h_in.cpu().numpy(), h_out.cpu().numpy()], axis=1)
                    n_nodes = h.shape[0]
                    h_o = h.reshape(n_nodes, 1, h.shape[1]).repeat(n_nodes, axis=1)
                    h_d = h.reshape(1, n_nodes, h.shape[1]).repeat(n_nodes, axis=0)
                    feat = np.concatenate([h_o, h_d, dis.reshape(n_nodes, n_nodes, 1)], axis=2)
                    od_hat = gbrt.predict(feat.reshape(-1, h.shape[1] * 2 + 1)).reshape(n_nodes, n_nodes)
                    od_hat[od_hat < 0] = 0
                metrics_all.append(cal_od_metrics(od_hat, od))
            del gmel, gbrt
            gc.collect()

        elif model_name == "NetGAN":
            model_dir = PROJECT_ROOT / "models" / model_name
            old_path = sys.path.copy()
            sys.path.insert(0, str(model_dir))
            try:
                spec = importlib.util.spec_from_file_location("netgan_model", model_dir / "model.py")
                model_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_mod)

                generator = model_mod.Generator().to(device)
                discriminator = model_mod.Discriminator().to(device)
                opt_g = torch.optim.Adam(generator.parameters(), lr=3e-4)
                opt_d = torch.optim.Adam(discriminator.parameters(), lr=3e-4)

                for epoch in tqdm(range(2), desc="NetGAN epochs"):
                    generator.train()
                    discriminator.train()
                    for nf, adj, dis, od in iter_graph_areas(train_areas, data_path):
                        nf_t = torch.FloatTensor(nfeat_scaler.transform(nf)).to(device)
                        dis_t = torch.FloatTensor(dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)).to(device)
                        graph = build_dgl_graph(adj, device)

                        opt_g.zero_grad()
                        fake_batch = generator.sample_generated_batch(graph, nf_t, dis_t, 128).to(device)
                        loss_g = -torch.mean(discriminator(fake_batch))
                        loss_g.backward()
                        opt_g.step()

                        if epoch % 5 == 0:
                            opt_d.zero_grad()
                            with torch.no_grad():
                                _, adjacency, logp = generator.generate_OD_net(graph, nf_t, dis_t)
                                batch = [model_mod.sample_one_random_walk(adjacency, logp) for _ in range(128)]
                                fake_batch = torch.stack(batch).to(device)
                            od_s = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
                            real_batch = torch.FloatTensor(model_mod.sample_batch_real(od_s)).to(device)
                            loss_d = (
                                torch.mean(discriminator(fake_batch))
                                - torch.mean(discriminator(real_batch))
                                + 10 * model_mod.compute_gradient_penalty(discriminator, real_batch, fake_batch)
                            )
                            loss_d.backward()
                            opt_d.step()

                generator.eval()
                for nf, adj, dis, od in tqdm(zip(nf_test, adj_test, dis_test, od_test), total=len(nf_test), desc="NetGAN eval"):
                    nf_t = torch.FloatTensor(nfeat_scaler.transform(nf)).to(device)
                    dis_t = torch.FloatTensor(dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)).to(device)
                    graph = build_dgl_graph(adj, device)
                    with torch.no_grad():
                        od_gen, _, _ = generator.generate_OD_net(graph, nf_t, dis_t)
                    od_hat = od_scaler.inverse_transform(od_gen.cpu().numpy())
                    od_hat[od_hat < 0] = 0
                    metrics_all.append(cal_od_metrics(od_hat, od))
                del generator, discriminator
                gc.collect()
            finally:
                sys.path = old_path
        else:
            raise ValueError(f"Unsupported graph model: {model_name}")
    finally:
        cleanup_gpu()

    avg = average_listed_metrics(metrics_all)
    print(f"  CPC={avg['CPC']:.4f}  RMSE={avg['RMSE']:.2f}  MAE={avg['MAE']:.2f}  ({time.time() - t0:.1f}s)")
    return metrics_all



def run_diffusion_model(model_name, train_areas, valid_areas, test_areas, data_path=DATA_PATH):
    del train_areas, valid_areas, test_areas, data_path
    print(f"\n{'=' * 60}\n  Running: {model_name}\n{'=' * 60}")
    t0 = time.time()
    model_dir = PROJECT_ROOT / "models" / model_name
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=model_dir,
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-300:]}")
            return []
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 2h")
        return []

    try:
        lines = result.stdout.strip().splitlines()
        dict_lines = []
        in_dict = False
        for line in lines:
            if line.strip().startswith("{"):
                in_dict = True
            if in_dict:
                dict_lines.append(line)
            if in_dict and line.strip().endswith("}"):
                break
        if dict_lines:
            avg_metrics = json.loads("\n".join(dict_lines).replace("'", '"'))
            print(f"  CPC={avg_metrics.get('CPC', 'N/A')}  ({time.time() - t0:.1f}s)")
            return [avg_metrics]
    except Exception as exc:
        print(f"  Could not parse metrics: {exc}")

    print(f"  Completed in {time.time() - t0:.1f}s (metrics not parsed)")
    return []
