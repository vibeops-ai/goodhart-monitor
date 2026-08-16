# GoodHart Monitor

The checker, built as a product: independent verification records for deployed
clinical prediction models. This repository contains the first end-to-end
maker/checker pair, built to open a research pilot on structured-data models.

## Why this use case

Two clinical AI leaders told us where to start, on the record:

- Tignanelli (UMN, chairs a health-system AI oversight committee): structured
  data is the frictionless entry — "readmission models, things like that...
  we could do it like tomorrow" — and the drift dashboard with review
  thresholds is the product "no one on the market" has; his team re-validates
  ~70+ models by hand quarterly.
- Singh (UCSD, chief health AI officer): the committee questions are local
  validation vs the vendor card, the work an alert stream creates (alert
  volume, PPV, patients-evaluated-per-true-case), and whether qualitative
  frontline signal matches the dashboard. His Epic sepsis paper is the
  canonical case: model card implied prediction; local measurement showed
  case-finding at AUROC 0.63 and 8 patients evaluated per true case.

## What is here

    data/                PhysioNet/CinC 2019 sepsis corpus (fetched, not vendored)
                         40,336 real ICU stays · two real hospital systems · Open Access
    scripts/fetch.sh     pulls the corpus
    scripts/build_matrices.py   psv -> causal per-hour features; A/B firewall by file layout
    scripts/train_maker.py      MAKER-1: ordinary vendor-grade sepsis model + its model card
    scripts/run_checker.py      THE PRODUCT: the four-section verification record
    out/record_GHM-0001.json    the record (content-addressed)
    PILOT.md             the research-pilot proposal this artifact exists to open

## The record's contract

Input: frozen model + its card + (features, adjudicated outcomes) from the
deployment population. Output: ACCEPTANCE / WORK / TIMING / DRIFT sections,
HOLDS·FAILS·INDETERMINATE per claim with bootstrap CIs, subgroups, and a
LIMITS section at the same weight as the findings. No PASS exists. Same
inputs, same record, byte for byte.

## Honesty rails carried over from the rest of the company

Real data only; anything constructed (the drift stream's ordering) is labelled
on the record itself. The maker is deliberately competent, not a straw man,
and every number on its card is true — the product's argument is that true
cards still fail locally, which is what Singh found in the wild.
