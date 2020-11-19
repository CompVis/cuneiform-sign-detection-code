### Generate aligned and placed detections
Given a trained sign detector these scripts perform the alignment and placement step of iterative training:

- `[iter = 0]`
    - `run_gen_initial_hypo.sh` : create initial placed detections by performing sign placement without aligned detections

- `[iter > 0]` (with aligned detections)
    - `run_iteration_sequential.sh` : perform alignment and placement step in one go
    
The generated raw, aligned and placed detections are stored in `results/results_ssd/[*model_version*]`.

To run the steps separately use the following scripts:

- `run_gen_detections.sh` : create raw detections by applying a trained sign detector to tablet images of train TL set
- `run_alignments.sh` : create aligned detections by performing line-level and sign-level alignment between raw detections and transliteration
- `run_cond_alignment.sh` : create placed detections by performing sign placement and combine them with the aligned detections


After aligned and placed detections have been generated, the sign detector training is performed using the notebooks in [experiments/sign_detector/](../../experiments/sign_detector/).

