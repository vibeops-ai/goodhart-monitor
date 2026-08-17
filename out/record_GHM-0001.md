# Verification record GHM-0001

**Subject** · MAKER-1 sepsis early warning 1.0.0  
**Deployment population** · hospital system B, 20,000 adult ICU stays, hourly scoring  
**Stream** · 761,995 rows · 20,000 entities · outcome prevalence 0.0141  
**Record hash** · `4cd3e90a9cb7e2be9bf7a8fe206605aa`

**Overall · FAILS**. Failing sections: acceptance, work

There is no PASS in this vocabulary. The good outcome names its scope.

## ACCEPTANCE · FAILS
_does the card's headline number hold on this population?_

Card says: **AUROC 0.81 for sepsis prediction** (on vendor dev set)

| | |
|---|---|
| measured here | 0.7564 |
| 95% interval (entity bootstrap) | 0.7423, 0.7685 |
| gap vs card | 0.0543 |
| tolerance allowed | 0.03 |
| rows / entities | 761,995 / 20,000 |

**Why this verdict** · measured 0.7564 against a floor of 0.7807 (card 0.8107 less 0.03 tolerance), and the interval's upper bound 0.7685 is still below it

## WORK · FAILS
_what work does the alert stream create at the shipped threshold?_

Card says: **~20% of alerts are true at the shipped threshold**

| | |
|---|---|
| threshold | 0.235347 |
| alerts per 100 entity-days | 34.4 |
| share of entities ever alerted | 0.024 |
| alert precision (row level) | 0.1083 |
| entities evaluated per actionable catch | 2.2 |
| …at sensitivity | 0.1944 |

> entities evaluated per actionable catch is the operational unit; the Epic sepsis model needed 8 at Michigan (Wong et al., JAMA Intern Med 2021). Only a first alert before onset counts as a catch. Read it beside the sensitivity: a threshold can buy a flattering ratio by refusing to alert, and no card mentions that trade

**Why this verdict** · alert precision 0.1083 against a floor of 0.1606 (0.8 of the card's 0.2007). Staffing consequence: 2.2 entities evaluated per actionable catch, at 19.4% sensitivity.

## TIMING · HOLDS
_does it warn before the event, or notice it afterwards?_

Card says: **predicts sepsis before clinical onset**

| | |
|---|---|
| positive entities | 1142 |
| alerted on at all | 289 |
| alerted before onset | 222 |
| share of catches after onset | 0.2318 |
| median lead when early (h) | 45 |
| within the 12h actionable window | 32 |
| …as a share of catches | 0.1107 |

| lead time | catches |
|---|---:|
| after onset | 67 |
| 0-6h | 16 |
| 6-12h | 16 |
| 12-24h | 37 |
| 24-48h | 52 |
| 48h+ | 101 |

> an alert that fires after onset is case finding, not prediction. Long leads are reported beside the actionable window because a warning two days early is experienced at the bedside as an unexplained alarm, and averaging the two hides that

**Why this verdict** · 222 of 289 catches (76.8%) preceded onset, against a policy floor of 50%. Separately, 32 (11.1%) landed inside the 12h actionable window; the rest were early enough that the bedside may not connect the alert to the deterioration

## DRIFT · INDETERMINATE
_windowed performance against explicit review thresholds_

Baseline: local acceptance measurement, never the vendor card (AUROC 0.7564, PPV 0.1083). Review triggers at a 0.05 AUROC drop or PPV below 0.5 of baseline.

| window | entities | AUROC | PPV | alerts | review |
|---|---:|---:|---:|---:|---|
| 1 | 2000 | 0.76 | 0.0872 | 1491 | no |
| 2 | 2000 | 0.6996 | 0.1066 | 788 | auroc |
| 3 | 2000 | 0.772 | 0.1295 | 896 | no |
| 4 | 2000 | 0.748 | 0.102 | 1088 | no |
| 5 | 2000 | 0.7599 | 0.1159 | 1018 | no |
| 6 | 2000 | 0.7365 | 0.078 | 1180 | no |
| 7 | 2000 | 0.7702 | 0.1009 | 1408 | no |
| 8 | 2000 | 0.7851 | 0.1077 | 854 | no |
| 9 | 2000 | 0.7659 | 0.1104 | 1123 | no |
| 10 | 2000 | 0.7635 | 0.161 | 1087 | no |

**1 of 10 windows trigger review.** Ordering: constructed from sorted entity id; the stream carries no calendar

> reported as an instrument check: it shows the review thresholds computing against a local baseline, on real scores, and says nothing about this deployment over time

**Why this verdict** · drift is a claim about time and this stream carries none. The windows above are cut from sorted entity ids, so the spread between them measures sampling variation between arbitrary groups of patients, not deterioration. To make this section answerable, supply the stream in calendar order and pass --ordered

## SUBGROUPS
_where is the number worse, and which dimensions cannot be asked?_

| dimension | group | entities | prevalence | AUROC |
|---|---|---:|---:|---:|
| Age | 50-64 | 6,278 | 0.014 | 0.7491 |
| Age | 65-79 | 6,582 | 0.0141 | 0.7754 |
| Age | 80+ | 2,322 | 0.0137 | 0.7852 |
| Age | <50 | 4,818 | 0.0146 | 0.7292 |
| Gender | recorded 0 | 9,268 | 0.0133 | 0.7634 |
| Gender | recorded 1 | 10,732 | 0.0149 | 0.75 |

AUROC spread across groups: 0.056

**Cannot be asked of this stream:** race, language, insurance. a dimension the stream does not carry is a finding, not a blank. If race, language or payer are absent, nobody can answer the equity question about this deployment, the vendor included

## CLAIMS THAT CANNOT BE TESTED

- **M-4** performance generalises to new hospital systems. _no supporting measurement provided_

> recorded, never scored. A claim with no number attached is not evidence, and its presence on a card is itself something the committee should weigh

## LIMITS

Carried at the same weight as the findings.

- outcome labels are whatever the supplied stream calls adjudicated; the checker does not re-adjudicate them and cannot detect a mislabelled outcome
- verification covers the population, threshold and window in this stream. It says nothing about any other population, threshold or window
- no PASS verdict exists. NO finding within scope is not the same claim as safe
- hospital B is a held-out system the maker never trained on; this record says nothing about hospital A

---
Re-run with the same inputs and this record hashes to `4cd3e90a9cb7e2be9bf7a8fe206605aa` again. If it does not, something changed and the difference is the finding.
