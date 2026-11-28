# External weekly features for the Zhejiang blood-shortage project

Built by `tools/build_weekly_features.py`. Each blood row's `date` is the **Monday that starts its week**
(confirmed: holiday weeks line up with collection drops when dates are week starts, correlation −0.25 vs −0.13).

## Files
| File | Contents |
|---|---|
| `weekly_external_features_2012_2026.csv` | One row per Monday, Mar 2012 – Dec 2026 (774 weeks), **including future weeks** for forecasting |
| `Zhejiang_blood_donation_corrected_enriched.csv` | **Recommended.** Corrected blood data (639 weeks) + all external features (639/639 weeks matched) + quarterly Data_Sheet_2 context |
| `Zhejiang_blood_donation_enriched.csv` | Same, built on the original file. 23 rows with mistyped dates match no Monday (blank features) and the mistyped October weeks get January's calendar, so prefer the corrected version |

The original 29 blood columns are unchanged in both enriched files; new columns are appended.

## Sources
1. **Public holidays and make-up workdays**: official State Council holiday notices, collected in
   https://github.com/NateScarlet/holiday-cn (each year file cites its gov.cn notice). Cross-checked against the
   `chinese_calendar` Python package: 0 mismatches over all weeks.
2. **COVID-19 restrictions**: Oxford COVID-19 Government Response Tracker (OxCGRT), China national
   `StringencyIndex_Average` (0–100), https://github.com/OxCGRT/covid-policy-dataset. Daily 2020–2022.
3. **Quarterly context** (GDP, health system, population): Data_Sheet_2 of the 2026 Frontiers in Public Health
   Zhejiang study, 2014–2023 (blank outside those years).

## Columns (per week starting on `date`)
| Column | Meaning |
|---|---|
| `holiday_days_off` | Official public-holiday days off in the week (0–7) |
| `makeup_workdays` | Weekend days that are official make-up workdays |
| `total_days_off` | Weekends + holidays − make-up workdays (normal week = 2) |
| `new_year_days`, `spring_festival_days`, `qingming_days`, `labour_day_days`, `dragon_boat_days`, `mid_autumn_days`, `national_day_days`, `other_holiday_days` | Days off for each holiday. When Mid-Autumn falls inside Golden Week (2012, 2017, 2020, 2023, 2025) the combined period counts as `national_day`. `other_holiday` = the one-off 3 Sep 2015 Victory Day holiday |
| `school_break_days_approx` | **Approximate** school-break days: 3 weeks either side of Spring Festival day 1, plus July–August. Not an official school calendar |
| `days_to_next_holiday`, `days_since_last_holiday` | Distance from the week's Monday to the nearest holiday day off |
| `covid_stringency_mean`, `covid_stringency_max` | OxCGRT stringency over the week. 0 before 2020; set to 0 from 2023 (China ended restrictions in Jan 2023; the tracker stops at 2022-12-31) |
| `next_week_*` | The calendar columns for the **following** week. Holidays are announced in advance, so these are legitimate inputs for forecasting week t+1 |
| `gdp_*`, `health_*`, `medical_staff`, `outpatient_er`, `population*` | Quarterly/annual Data_Sheet_2 context (see earlier notes: use the previous quarter's GDP for forecasting) |

## Leakage notes
* Calendar columns (holidays, school breaks) are known in advance, so current and `next_week_*` values are safe.
* COVID stringency is observed during the week, so use the current week's value (not next week's) when forecasting.

## Descriptive link to shortage risk (corrected data, all labeled weeks; not a model result)
| Next week has… | High | Medium | Low |
|---|---:|---:|---:|
| no holiday days off (507 weeks) | 15.3% | 46.4% | 38.4% |
| 1–2 holiday days off (69 weeks) | 15.1% | 50.9% | 34.1% |
| **3–7 holiday days off (56 weeks)** | **30.8%** | 45.8% | 23.3% |
| full approximate school-break week (158 weeks) | 22.2% | 46.1% | 31.7% |

Long holidays roughly double the chance of a HIGH-risk week. The model already knows Spring Festival timing,
so part of this effect is already used; National Day, Labour Day and the other holidays are new information.

## Not obtained
Weather (Hangzhou temperature, rain, typhoons) and weekly flu activity (WHO FluNet) could not be downloaded from
this environment. Both are free: Open-Meteo historical weather (https://open-meteo.com, Hangzhou 30.27N 120.15E)
and WHO FluNet (China). If you download them, the same weekly merge can add them.
