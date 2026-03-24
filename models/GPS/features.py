from .config import USE_LU_FEATURES, USE_JOBS_FEATURES

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

SELECTED_DEMO = ["pop_total","pop_male","pop_female","median_age_total","median_age_male","median_age_female","mean_travel_time_min"]
SELECTED_POI = ALL_POI_FEATURE_NAMES
SELECTED_LU = ALL_LU_FEATURE_NAMES
SELECTED_JOBS = ALL_JOBS_FEATURE_NAMES

DEMO_COL_IDX = [ALL_DEMO_FEATURE_NAMES.index(n) for n in SELECTED_DEMO]
POI_COL_IDX = [ALL_POI_FEATURE_NAMES.index(n) for n in SELECTED_POI]
LU_COL_IDX = [ALL_LU_FEATURE_NAMES.index(n) for n in SELECTED_LU]
JOBS_COL_IDX = [ALL_JOBS_FEATURE_NAMES.index(n) for n in SELECTED_JOBS]

FEATURE_NAMES = SELECTED_DEMO + SELECTED_POI
if USE_LU_FEATURES:
    FEATURE_NAMES += SELECTED_LU
if USE_JOBS_FEATURES:
    FEATURE_NAMES += SELECTED_JOBS
