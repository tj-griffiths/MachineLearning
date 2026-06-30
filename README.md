# Machine Learning Projects — Particle Physics

**Thomas Griffiths** — Data Analysis and Machine Learning, University of Edinburgh

Two independent notebooks applying machine learning to LHC-style collider data:
an unsupervised **anomaly detector** for finding new physics without labeled
BSM examples, and a supervised **binary classifier** used to sharpen a real
search for a heavy resonance. Both work with simulated detector-level kinematics
(jets, leptons, missing energy) and both ultimately ask the same question —
can ML separate "interesting" events from Standard Model background better
than hand-picked cuts — but from opposite ends: one with no signal labels at
all, one with full signal/background truth.

## Projects

### `BSM_Search.ipynb` — Anomaly Detection for Exotic Event Identification

Trains an **autoencoder** on Standard Model (SM) events only, then uses
reconstruction error as an anomaly score to flag nine different
Beyond-Standard-Model (BSM) signal models (SUSY chargino/neutralino
production, $Z'$ decays, gluino/RPV-SUSY) — without the network ever seeing
BSM data during training. The model-independent strategy mirrors how a real
search for *unknown* new physics would have to work.

- **Headline result:** the Gluino and Stlp (RPV-SUSY-like) models separate
  clearly from SM (~4σ / ~3σ); the SUSY chargino/neutralino and $Z'$ models
  tested look largely SM-like to this model.
- Full details, preprocessing steps, architecture, and known limitations:
  see [`BSM_Search.ipynb`](BSM_Search.ipynb) — the notebook is self-contained
  with inline documentation throughout.

### `ATLAS_VZ.ipynb` — Exotic Searches with ATLAS and ML Classification

A full search analysis for a hypothetical 1 TeV heavy Higgs decaying via
$H \to ZV$ ($Z \to \ell^+\ell^-$, $V \to qq'$ as a boosted fat-jet), benchmarked
against three SM background processes (diboson, Z+jets, top). The analysis
runs the **same search twice**: once with traditional rectangular kinematic
cuts and a binned $\chi^2$ fit, once with a supervised **neural network
classifier** trained to separate signal from background, each followed by a
Wilks' theorem significance calculation so the two approaches can be compared
directly. A final section deliberately includes the search observable
(`reco_zv_mass`) as a training feature to demonstrate the data leakage and
background-sculpting this causes.

- **Headline result:** rectangular cuts alone give a ~6.83σ discovery
  significance; adding the NN classifier (trained without the mass variable)
  raises this to ~14.24σ. Including the mass variable as a training feature
  further boosts the *apparent* S/B but invalidates the search by sculpting
  the background to mimic the signal peak.
- Full details, physics model, fit procedure, and discussion: see
  [`ATLAS_VZ.ipynb`](ATLAS_VZ.ipynb).

## Shared context

Both notebooks deal with the same category of LHC event data: per-event
kinematics (energy, $p_T$, $\eta$, $\phi$) for reconstructed particles, plus
missing transverse energy (MET) as a proxy for invisible particles like
neutrinos. Both also use PyTorch for their respective neural network
components and a SM-events-only training philosophy, just applied to
different problems — unsupervised anomaly detection in `BSM_Search.ipynb`
versus supervised classification in `ATLAS_VZ.ipynb`.

## Running

Each notebook expects its own dataset locally (`data/Proj2/...` for
`BSM_Search.ipynb`, `data/W18_Proj4/...` for `ATLAS_VZ.ipynb` — see each
notebook's data-loading cell for exact filenames). Requirements across both:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`, `scikit-learn`, `scipy`,
`tqdm`. Run each notebook top to bottom independently; they don't share state.

## Known limitations

See each notebook's own limitations discussion for project-specific issues
(e.g. `BSM_Search.ipynb` has a particle-counting bug that zeroes out two of
its input features — documented inline). 

## Acknowledgements

`BSM_Search.ipynb`: *DAML Project 2 — Anomaly Detection for Exotic Event
Identification at the Large Hadron Collider*, Robert Currie, University of
Edinburgh. Background and BSM datasets from *The Dark Machines Anomaly Score
Challenge* (arXiv:2105.14027).

`ATLAS_VZ.ipynb`: *DAML Project 4 — Exotic Searches with ATLAS and ML
Classification*, Christos Leonidopoulos, University of Edinburgh, March 2026.