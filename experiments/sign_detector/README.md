### Perform sign detector training
After the sign annotations (aligned and placed detections) have been generated and stored under `results/results_ssd/` using the scripts in `scripts/generate/`,
the sign detector is trained by performing the following steps:

1) use `train_sign_classifier.ipynb` as template to train sign classifier 
2) use `train_sign_detector.ipynb` as template to train sign detector (initialized with pre-trained sign classifier from 1.)
3) in semi-supervised case, use `finetune_sign_detector.ipynb` to fine-tune sign detector on manual annotations

### Eval sign detector

- use `test_sign_detector.ipynb` for evaluation of the sign detector
