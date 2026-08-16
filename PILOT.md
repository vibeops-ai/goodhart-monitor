# Pilot proposal · University of Minnesota · structured-data model verification

**For:** Dr. Chris Tignanelli · chair, health-system AI oversight committee
**From:** GoodHart Labs
**Shape:** research project under the health system's umbrella, IRB as required.
Structured data first, because you told us that path opens "tomorrow."

## What we built to earn this meeting

You said the easiest starting point is discrete-data models: readmission,
deterioration, sepsis. So we built the whole loop once, end to end, on real
data, and brought the output rather than a deck.

- **The maker.** A sepsis early-warning model of the ordinary commercial kind:
  gradient-boosted trees, hourly vitals and labs, trained on one real hospital
  system (PhysioNet/CinC 2019, hospital A, ~20k ICU stays). It ships with a
  model card whose every number is true, measured by the maker, on the
  maker's own data, phrased the way vendors phrase it.
- **The checker.** The GoodHart verification record, run where the model is
  "deployed": hospital system B, 20k real ICU stays the maker never saw.

The checker never sees the model. It reads a scored stream: `entity_id`,
`score`, `label`, and a timestamp if one exists. That is what your EHR
already writes every time the model fires. That is a deliberate design
constraint rather than a convenience: a verifier that needs the vendor's
artifact is a verifier the vendor can decline.

The record has five sections, and each one is a question your committee
already asks:

| Section | The question | Whose question it is |
|---|---|---|
| ACCEPTANCE | does the card's number hold on this population? | Singh: "come with your own data, not the vendor's" |
| WORK | alerts/day, PPV, patients evaluated per true case | Singh: "what work does this model create, and is it valuable?" |
| TIMING | does it warn before onset, or notice afterwards? | the exact claim class the Epic sepsis card got wrong |
| DRIFT | windowed performance against explicit review thresholds | yours: "5% AUROC deterioration triggers review, that's missing; we do it manually every three months" |
| SUBGROUPS | where is it worse, and which dimensions cannot be asked at all? | the equity question, answered honestly: a dimension the stream does not carry is reported as a gap, not left blank |

Verdicts are HOLDS / FAILS / INDETERMINATE per claim, with bootstrap
intervals. There is no PASS anywhere: the good outcome names its scope. The
record carries a LIMITS section at the same weight as the findings, and the
whole thing is content-addressed. Same inputs, same record, byte for byte,
re-runnable by your analysts without us in the room.

## What the pilot would be

One of your committee's already-deployed structured-data models, Epic sepsis,
deterioration, or a readmission model, under the research umbrella:

1. **Inputs we need.** One retrospective export: patient identifier, the
   model's score, the adjudicated outcome, a timestamp, and whichever
   demographic columns your committee wants the equity cut on. Nothing else.
   No model, no features, no vendor cooperation, because the scores are
   already logged, and you said the structured path needs no vendor negotiation
   at all.
2. **What we produce.** The four-section record above, on your population,
   with your committee's thresholds substituted for our defaults, plus the
   windowed monitor your analysts currently reproduce by hand every quarter.
3. **Your thresholds, not ours.** Every number that produces a verdict lives
   in a config file the committee owns and the record prints: what counts as a
   deterioration worth reviewing, how far below a card's number still holds,
   what lead time is actually actionable for your bundle. We ship defaults
   taken from what you and Singh said out loud; you overwrite them.
4. **What it costs you.** Data access under the research protocol and a
   clinical reviewer for the misclassification sample. We bring everything
   else.
5. **Publication.** The record format and the local-validation results are
   publishable in exactly the lane your Epic deterioration work sits in; we
   want the method paper, you take clinical lead.

## What this is not

We do not train, tune or repair the model. The maker stays the vendor's. A
verifier that also sells fixes is grading its own homework, which is the
disease this product exists to treat. And nothing in the pilot touches
clinical care: retrospective scores, research umbrella, no CDS in the loop,
which is also why the FDA question does not gate this phase.

## What the record said about our own maker

Every number on that card is true. Measured on hospital B:

| Section | Verdict | Why |
|---|---|---|
| ACCEPTANCE | FAILS | AUROC 0.7564 (95% CI 0.7423–0.7685, cluster bootstrap over patients) against a card claim of 0.8107. The interval's upper bound does not reach the floor. |
| WORK | FAILS | 10.8% of alert-hours are true against a 20.1% claim; 34.4 alerts per 100 patient-days; 2.2 patients evaluated per actionable catch, but only at 19.4% sensitivity, so the ratio looks good because the threshold rarely fires. |
| TIMING | HOLDS | 77% of catches precede onset. The Epic failure did not reproduce here, and the record says so. But only 11% land inside a 12h actionable window; the median early warning is 45h ahead, which the bedside experiences as an unexplained alarm. |
| DRIFT | INDETERMINATE | The public corpus carries no calendar, so there is nothing to measure drift against. The windows are printed as an instrument check and the record states what would make the section answerable. |

Two of those are worth more than the failures. TIMING holding is the checker
declining to find the fashionable failure when it is not there. DRIFT refusing
to return a verdict is the checker declining to sell you the exact dashboard
you asked for, on data that cannot support it. A verifier that always finds
something is a verifier you stop reading.

## The one number to remember

The maker's card is entirely true, and it still fails locally. That is not an
indictment of the maker. It is the argument that the card and the deployment
are different objects, and only one of them is being measured today.
