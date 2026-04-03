import json
import torch

from models.GPS.config import WEIGHTS_DIR, device
from models.GPS.data_load import prepare_multi_city_data, prepare_single_city_data
from models.GPS.lgbm_pipeline import load_lgbm_results as load_saved_lgbm_results
from models.GPS.metrics import cal_od_metrics, predict_full_matrix
from models.GPS.model import make_model

from .config import DATA_PATH, MULTI_CITY_IDS, SINGLE_CITY_ID, cleanup_gpu

_PE_TYPE_UNSET = object()


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

    def load_gps_results(self, run_id, city_data=None, config=None, area_id=None):
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

        print(f"  Loading {run_id} (pe_type={effective_cfg.pe_type}) ...")
        model = None
        try:
            model = make_model(effective_cfg, graph_data_ref=city_data["graph_data"])
            model.load_state_dict(torch.load(str(weight_path), map_location=device))
            model.to(device).eval()
            with torch.no_grad():
                pred = predict_full_matrix(model, city_data, effective_cfg)
            pred[pred < 0] = 0
            metrics = cal_od_metrics(pred, city_data["od_matrix_np"])
            print(f"  {run_id}: CPC={metrics['CPC']:.4f}  MAE={metrics['MAE']:.4f}")
            return metrics
        except Exception as exc:
            print(f"  ERROR loading {run_id}: {exc}")
            return None
        finally:
            if model is not None:
                del model
            cleanup_gpu()

    def load_lgbm_results(self, run_id, city_data=None, area_id=None, pe_type=_PE_TYPE_UNSET):
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
        return load_saved_lgbm_results(run_id, city_data)

    def load_multi_city_gps_results(self, run_id, city_ids=None):
        from models.GPS.config import load_model_config

        saved_cfg = load_model_config(run_id)
        if saved_cfg is None:
            print(f"  [SKIP] {run_id}: config JSON not found.")
            return []
        city_data_dict, _, _, _ = self.get_multi_city_data(pe_type=saved_cfg.pe_type, city_ids=city_ids)
        metrics = []
        for city_id in city_data_dict:
            city_metric = self.load_gps_results(run_id, city_data=city_data_dict[city_id], config=saved_cfg)
            if city_metric:
                metrics.append(city_metric)
        return metrics
