# Pilot proposal · University of Minnesota · structured-data model verification

**For:** Dr. Chris Tignanelli · chair, health-system AI oversight committee
**From:** GoodHart Labs
**Shape:** research project under the health system's umbrella, IRB as required.
Structured data first, because you told us that path opens "tomorrow."

## What we built to earn this meeting

You said the easiest starting point is discrete-data models — readmission,
deterioration, sepsis. So we built the whole loop once, end to end, on real
data, and brought the output rather than a deck.

- **The maker.** A sepsis early-warning model of the ordinary commercial kind:
  gradient-boosted trees, hourly vitals and labs, trained on one real hospital
  system (PhysioNet/CinC 2019, hospital A, ~20k ICU stays). It ships with a
  model card whose every number is true — measured by the maker, on the
  maker's own data, phrased the way vendors phrase it.
- **The checker.** The GoodHart verification record, run where the model is
  "deployed": hospital system B, 20k real ICU stays the maker never saw.

The record has four sections, and each one is a question your committee
already asks:

| Section | The question | Whose question it is |
|---|---|---|
| ACCEPTANCE | does the card's number hold on this population? | Singh: "come with your own data, not the vendor's" |
| WORK | alerts/day, PPV, patients evaluated per true case | Singh: "what work does this model create, and is it valuable?" |
| TIMING | does it warn before onset, or notice afterwards? | the exact claim class the Epic sepsis card got wrong |
| DRIFT | windowed performance against explicit review thresholds | yours: "5% AUROC deterioration triggers review — that's missing; we do it manually every three months" |

Verdicts are HOLDS / FAILS / INDETERMINATE per claim, with bootstrap
intervals. There is no PASS anywhere: the good outcome names its scope. The
record carries a LIMITS section at the same weight as the findings, and the
whole thing is content-addressed — same inputs, same record, byte for byte,
re-runnable by your analysts without us in the room.

## What the pilot would be

One of your committee's already-deployed structured-data models — Epic sepsis,
deterioration, or a readmission model — under the research umbrella:

1. **Inputs we need.** Retrospective: the model's scores (or access to score
   the features) and the adjudicated outcomes, for a defined window. No
   vendor cooperation required if scores are already logged in Epic; you said
   the structured path needs no vendor negotiation at all.
2. **What we produce.** The four-section record above, on your population,
   with your committee's thresholds substituted for our defaults — plus the
   windowed monitor your analysts currently reproduce by hand every quarter.
3. **What it costs you.** Data access under the research protocol and a
   clinical reviewer for the misclassification sample. We bring everything
   else.
4. **Publication.** The record format and the local-validation results are
   publishable in exactly the lane your Epic deterioration work sits in; we
   want the method paper, you take clinical lead.

## What this is not

We do not train, tune or repair the model. The maker stays the vendor's. A
verifier that also sells fixes is grading its own homework, which is the
disease this product exists to treat. And nothing in the pilot touches
clinical care: retrospective scores, research umbrella, no CDS in the loop —
which is also why the FDA question does not gate this phase.

## The one number to remember

On our own build: the maker's card is entirely true, and it still fails
locally. That is not an indictment of the maker. It is the argument that the
card and the deployment are different objects, and only one of them is being
measured today.
