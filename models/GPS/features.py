import numpy as np

from .config import FEATURE_PRESET, USE_LU_FEATURES, USE_JOBS_FEATURES

ALL_DEMO_FEATURE_NAMES = [
    "pop_total","pop_male","pop_female",
    "age_under5_total","age_under5_male","age_under5_female",
    "age_5to9_total","age_5to9_male","age_5to9_female",
    "age_10to14_total","age_10to14_male","age_10to14_female",
    "age_15to19_total","age_15to19_male","age_15to19_female",
    "age_20to24_total","age_20to24_male","age_20to24_female",
    "age_25to29_total","age_25to29_male","age_25to29_female",
    "age_30to34_total","age_30to34_male","age_30to34_female",
    "age_35to39_total","age_35to39_male","age_35to39_female",
    "age_40to44_total","age_40to44_male","age_40to44_female",
    "age_45to49_total","age_45to49_male","age_45to49_female",
    "age_50to54_total","age_50to54_male","age_50to54_female",
    "age_55to59_total","age_55to59_male","age_55to59_female",
    "age_60to64_total","age_60to64_male","age_60to64_female",
    "age_65to69_total","age_65to69_male","age_65to69_female",
    "age_70to74_total","age_70to74_male","age_70to74_female",
    "age_75to79_total","age_75to79_male","age_75to79_female",
    "age_80to84_total","age_80to84_male","age_80to84_female",
    "age_85plus_total","age_85plus_male","age_85plus_female",
    "median_age_total","median_age_male","median_age_female","median_earnings",
    "worker_private_wage","worker_government","worker_self_employed","worker_unpaid_family",
    "mean_travel_time_min",
    "vehicles_none","vehicles_1","vehicles_2","vehicles_3plus",
    "total_households","avg_household_size","total_families","avg_family_size",
    "enroll_nursery_preschool","enroll_k12","enroll_kindergarten",
    "enroll_elem_grade1to4","enroll_elem_grade5to8",
    "enroll_hs_grade9to12","enroll_college_undergrad","enroll_grad_professional",
    "edu_9to12_no_diploma","edu_associates","edu_bachelors",
    "edu_bachelors_or_higher","edu_grad_professional",
    "edu_hs_graduate","edu_hs_or_higher","edu_less_than_9th","edu_less_than_hs",
    "edu_25to34_bachelors_or_higher","edu_25to34_hs_or_higher",
    "edu_some_college_or_associates","edu_some_college_no_degree",
    "poverty_male","poverty_female"]

ALL_POI_FEATURE_NAMES = [
    "poi_finance","poi_public","poi_transport","poi_entertainment","poi_health",
    "poi_service","poi_education","poi_government","poi_religion",
    "poi_accommodation","poi_food","poi_cafe","poi_fast_food","poi_ice_cream",
    "poi_pub","poi_restaurant","poi_shop_beauty","poi_shop_clothes",
    "poi_boutique","poi_shop_transport","poi_retail","poi_commodity",
    "poi_marketplace","poi_home_improvement","poi_sport","poi_public_transport",
    "poi_kindergarten","poi_office","poi_recycling","poi_travel_agency",
    "poi_tourism","poi_shop_livelihood","poi_residential","poi_dormitory"]

ALL_LU_FEATURE_NAMES = ["residential","business","recreation","industrial","transport","special","agriculture"]
ALL_JOBS_FEATURE_NAMES = ["jobs_total"]

assert len(ALL_DEMO_FEATURE_NAMES) == 97
assert len(ALL_POI_FEATURE_NAMES) == 34

REDUCED_DEMO_FEATURE_NAMES = [
    "pop_total","pop_male","pop_female",
    "median_age_total","median_age_male","median_age_female",
    "mean_travel_time_min",
]

FEATURE_PRESET_OPTIONS = ("all", "reduced")


def _normalize_feature_preset(feature_preset=None):
    feature_preset = FEATURE_PRESET if feature_preset is None else feature_preset
    if feature_preset not in FEATURE_PRESET_OPTIONS:
        raise ValueError(
            f"Unknown feature_preset={feature_preset!r}. "
            f"Valid options: {FEATURE_PRESET_OPTIONS}"
        )
    return feature_preset


def get_feature_spec(feature_preset=None, use_lu=USE_LU_FEATURES, use_jobs=USE_JOBS_FEATURES):
    feature_preset = _normalize_feature_preset(feature_preset)

    selected_demo = (
        list(ALL_DEMO_FEATURE_NAMES)
        if feature_preset == "all"
        else list(REDUCED_DEMO_FEATURE_NAMES)
    )
    selected_poi = list(ALL_POI_FEATURE_NAMES)
    selected_lu = list(ALL_LU_FEATURE_NAMES)
    selected_jobs = list(ALL_JOBS_FEATURE_NAMES)

    feature_names = selected_demo + selected_poi
    if use_lu:
        feature_names += selected_lu
    if use_jobs:
        feature_names += selected_jobs

    return {
        "feature_preset": feature_preset,
        "selected_demo": selected_demo,
        "selected_poi": selected_poi,
        "selected_lu": selected_lu,
        "selected_jobs": selected_jobs,
        "demo_idx": [ALL_DEMO_FEATURE_NAMES.index(name) for name in selected_demo],
        "poi_idx": [ALL_POI_FEATURE_NAMES.index(name) for name in selected_poi],
        "lu_idx": [ALL_LU_FEATURE_NAMES.index(name) for name in selected_lu],
        "jobs_idx": [ALL_JOBS_FEATURE_NAMES.index(name) for name in selected_jobs],
        "feature_names": feature_names,
    }


def build_feature_matrix(raw, feature_preset=None, use_lu=USE_LU_FEATURES, use_jobs=USE_JOBS_FEATURES):
    spec = get_feature_spec(feature_preset=feature_preset, use_lu=use_lu, use_jobs=use_jobs)
    parts = [
        raw["demos"][:, spec["demo_idx"]],
        raw["pois"][:, spec["poi_idx"]],
    ]
    if use_lu and raw.get("lu") is not None:
        parts.append(raw["lu"][:, spec["lu_idx"]])
    if use_jobs and raw.get("jobs") is not None:
        parts.append(raw["jobs"][:, spec["jobs_idx"]])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


ACTIVE_FEATURE_SPEC = get_feature_spec()
SELECTED_DEMO = ACTIVE_FEATURE_SPEC["selected_demo"]
SELECTED_POI = ACTIVE_FEATURE_SPEC["selected_poi"]
SELECTED_LU = ACTIVE_FEATURE_SPEC["selected_lu"]
SELECTED_JOBS = ACTIVE_FEATURE_SPEC["selected_jobs"]

DEMO_COL_IDX = ACTIVE_FEATURE_SPEC["demo_idx"]
POI_COL_IDX = ACTIVE_FEATURE_SPEC["poi_idx"]
LU_COL_IDX = ACTIVE_FEATURE_SPEC["lu_idx"]
JOBS_COL_IDX = ACTIVE_FEATURE_SPEC["jobs_idx"]
FEATURE_NAMES = ACTIVE_FEATURE_SPEC["feature_names"]
