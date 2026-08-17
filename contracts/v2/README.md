# Contracts v2.0.0

`goodhart-verifier-contracts.schema.json` is the canonical JSON Schema 2020-12
package for cross-module messages. It is copied verbatim from the product
documentation set and is not edited here. If the canonical artifact changes,
this file is replaced and every emitter is re-validated against it.

Objects, and who writes them in this repository:

| Object | Writer here |
|---|---|
| `ClaimContract` | onboarding (`contracts.py`, from the vendor card) |
| `AuthorityBundle` | onboarding (the corpus's Sepsis-3 label definition) |
| `FailureModeProfile` | onboarding |
| `VerificationPolicy` | onboarding (from `governance.toml`) |
| `VerificationEvent` | the runtime adapter, one per scored patient-hour in scope |
| `CheckResult` | the check packs |
| `Verdict` | the composer |
| `DispositionEvent` | the router |
| `TruthResolution` | the resolver, once the outcome window matures |
| `PeriodicReport` | monitoring |

Every object this repository emits is validated against the schema before it is
written. A failing object is a build failure, not a warning.
