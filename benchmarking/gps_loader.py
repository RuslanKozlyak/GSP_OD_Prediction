import json
import torch

from models.GPS.config import WEIGHTS_DIR, device
from models.GPS.data_load import prepare_multi_city_data, prepare_single_city_data
from models.GPS.lgbm_pipeline import load_lgbm_results as load_saved_lgbm_results
from models.shared.metrics import cal_od_metrics, compute_metrics
from models.GPS.metrics import predict_full_matrix
from models.GPS.model import make_model

from .config import DATA_PATH, MULTI_CITY_IDS, SINGLE_CITY_ID, cleanup_gpu
from .config import set_global_seed

_PE_TYPE_UNSET = object()


def _enrich_metrics(metrics, pred, city_data, is_test_city=True):
    """Add benchmark-specific test metrics to a cal_od_metrics dict."""
    import numpy as np
    od = city_data['od_matrix_np']
    if city_data.get('split_scope') == 'multi_city':
        if is_test_city:
            metrics['CPC_test'] = metrics['CPC']
            metrics['MAE_test'] = metrics['MAE']
            metrics['RMSE_test'] = metrics['RMSE']
        else:
            metrics['CPC_test'] = np.nan
            metrics['MAE_test'] = np.nan
            metrics['RMSE_test'] = np.nan
    else:
        tm = city_data.get('test_mask')
        if tm is not None and np.any(tm):
            mt = compute_metrics(pred[tm], od[tm].astype(float))
            metrics['CPC_test'] = mt['CPC']
            metrics['MAE_test'] = mt['MAE']
            metrics['RMSE_test'] = mt['RMSE']


class GPSBenchmarkLoader:
    def __init__(self, single_city_id=SINGLE_CITY_ID, multi_city_ids=None, data_path=DATA_PATH):
        self.single_city_id = single_city_id
        self.multi_city_ids = list(multi_city_ids or MULTI_CITY_IDS)
        self.data_path = data_path
        self._single_city_cache = {}
        self._multi_city_cache = {}

    def get_single_city_data(self, pe_type="rwpe", area_id=None):
        area_id = area_id or self.single_city_id
        key = (area_id, pe_type)
        if key not in self._single_city_cache:
            self._single_city_cache[key] = prepare_single_city_data(
                area_id=area_id,
                pe_type=pe_type,
                data_path=str(self.data_path),
            )
        return self._single_city_cache[key]

    def get_multi_city_data(self, pe_type="rwpe", city_ids=None):
        city_ids = tuple(city_ids or self.multi_city_ids)
        key = (city_ids, pe_type)
        if key not in self._multi_city_cache:
            self._multi_city_cache[key] = prepare_multi_city_data(
                city_ids=list(city_ids),
                pe_type=pe_type,
                data_path=str(self.data_path),
            )
        return self._multi_city_cache[key]

    def load_gps_results(self, run_id, city_data=None, config=None, area_id=None,
                         inference_seed=None, verbose=True, is_test_city=True):
        from models.GPS.config import load_model_config

        weight_path = WEIGHTS_DIR / f"{run_id}.pt"
        if not weight_path.exists():
            print(f"  [SKIP] {run_id}: weights not found at {weight_path}")
            return None

        saved_cfg = load_model_config(run_id)
        effective_cfg = saved_cfg if saved_cfg is not None else config
        if effective_cfg is None:
            print(f"  [SKIP] {run_id}: no saved config JSON and no config passed.")
            return None

        if city_data is None:
            city_data = self.get_single_city_data(pe_type=effective_cfg.pe_type, area_id=area_id)

        if verbose:
            print(f"  Loading {run_id} (pe_type={effective_cfg.pe_type}) ...")
        model = None
        try:
            if inference_seed is not None:
                set_global_seed(inference_seed)
            model = make_model(effective_cfg, graph_data_ref=city_data["graph_data"])
            model.load_state_dict(torch.load(str(weight_path), map_location=device))
            model.to(device).eval()
            with torch.no_grad():
                pred = predict_full_matrix(model, city_data, effective_cfg)
            pred[pred < 0] = 0
            metrics = cal_od_metrics(pred, city_data["od_matrix_np"])
            _enrich_metrics(metrics, pred, city_data, is_test_city=is_test_city)
            if verbose:
                print(f"  {run_id}: CPC={metrics['CPC']:.4f}  MAE={metrics['MAE']:.4f}")
            return metrics
        except Exception as exc:
            print(f"  ERROR loading {run_id}: {exc}")
            return None
        finally:
            if model is not None:
                del model
            cleanup_gpu()

    def load_lgbm_results(self, run_id, city_data=None, area_id=None, pe_type=_PE_TYPE_UNSET, inference_seed=None):
        from models.GPS.config import load_model_config

        if city_data is None:
            donor_id = None
            meta_path = WEIGHTS_DIR / f"{run_id}_meta.json"
            if meta_path.exists():
                donor_id = json.loads(meta_path.read_text()).get("donor_id")
            if donor_id is None and run_id.endswith("_lgbm"):
                donor_id = run_id[:-5]
            donor_cfg = load_model_config(donor_id) if donor_id else None
            effective_pe_type = pe_type if pe_type is not _PE_TYPE_UNSET else (
                donor_cfg.pe_type if donor_cfg is not None else "rwpe"
            )
            city_data = self.get_single_city_data(pe_type=effective_pe_type, area_id=area_id)
        if inference_seed is not None:
            set_global_seed(inference_seed)
        return load_saved_lgbm_results(run_id, city_data)

    def load_gmel_gps_results(self, run_id, city_data=None, area_id=None, inference_seed=None):
        """Load a pre-trained GMEL_GPS model + GBRT and evaluate on one city."""
        import json
        from models.GMEL_GPS.model import GMEL_GPS
        from models.GPS.config import TrainingConfig
        from models.GMEL_GPS.main import predict_gmel_gps

        weight_path = WEIGHTS_DIR / f"{run_id}.pt"
        gbrt_path   = WEIGHTS_DIR / f"{run_id}_gbrt.joblib"
        lgbm_path   = WEIGHTS_DIR / f"{run_id}_lgbm.lgbm"
        cfg_path    = WEIGHTS_DIR / f"{run_id}.json"

        decoder_path = lgbm_path if lgbm_path.exists() else gbrt_path
        if not weight_path.exists() or not decoder_path.exists():
            print(f"  [SKIP] {run_id}: weights or decoder not found")
            return None

        with open(cfg_path) as f:
            raw = json.load(f)
        # Filter out fields not in TrainingConfig
        valid_fields = {f.name for f in __import__('dataclasses').fields(TrainingConfig)}
        cfg = TrainingConfig(**{k: v for k, v in raw.items() if k in valid_fields})

        if city_data is None:
            city_data = self.get_single_city_data(
                pe_type=cfg.pe_type, area_id=area_id
            )

        print(f"  Loading {run_id} (pe_type={cfg.pe_type}) ...")
        model = None
        try:
            if inference_seed is not None:
                set_global_seed(inference_seed)
            gd = city_data['graph_data']
            model = GMEL_GPS(
                input_dim  = gd.x.shape[1],
                edge_dim   = gd.edge_attr.shape[1],
                pe_type    = cfg.pe_type,
                norm_type  = cfg.gps_norm_type,
            ).to(device)
            model.load_state_dict(
                torch.load(str(weight_path), map_location=device)
            )
            if cfg.decoder_type == 'lgbm' and lgbm_path.exists():
                import lightgbm as lgb
                decoder = lgb.Booster(model_file=str(lgbm_path))
            else:
                import joblib
                decoder = joblib.load(str(gbrt_path))
            pred = predict_gmel_gps(model, decoder, city_data, device)
            metrics = cal_od_metrics(pred, city_data['od_matrix_np'])
            _enrich_metrics(metrics, pred, city_data)
            print(f"  {run_id}: CPC={metrics['CPC']:.4f}  MAE={metrics['MAE']:.4f}")
            return metrics
        except Exception as exc:
            print(f"  ERROR loading {run_id}: {exc}")
            return None
        finally:
            if model is not None:
                del model
            cleanup_gpu()

    def load_multi_city_gps_results(self, run_id, city_ids=None, inference_seed=None,
                                    evaluate_all_cities=False, return_split_groups=False,
                                    verbose=True):
        from models.GPS.config import load_model_config

        saved_cfg = load_model_config(run_id)
        if saved_cfg is None:
            print(f"  [SKIP] {run_id}: config JSON not found.")
            return []
        city_data_dict, _, _, test_city_ids = self.get_multi_city_data(
            pe_type=saved_cfg.pe_type, city_ids=city_ids,
        )
        test_city_ids = list(test_city_ids)
        test_city_id_set = set(test_city_ids)
        eval_city_ids = list(city_data_dict.keys()) if evaluate_all_cities else list(test_city_ids)
        metrics = []
        test_metrics = []
        for city_id in eval_city_ids:
            city_metric = self.load_gps_results(
                run_id,
                city_data=city_data_dict[city_id],
                config=saved_cfg,
                inference_seed=inference_seed,
                verbose=verbose,
                is_test_city=city_id in test_city_id_set,
            )
            if city_metric:
                city_metric = dict(city_metric)
                city_metric["city_id"] = city_id
                city_metric["is_test_city"] = city_id in test_city_id_set
                metrics.append(city_metric)
                if city_id in test_city_id_set:
                    test_metrics.append(city_metric)
        if return_split_groups:
            return {"all": metrics, "test": test_metrics}
        return metrics
