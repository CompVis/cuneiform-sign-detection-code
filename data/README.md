### Data folder

Place [*cuneiform-sign-detection-dataset*](https://github.com/to3i/cuneiform-sign-detection-dataset)  folders here:
- ./data/annotations
- ./data/images
- ./data/segments
- ./data/transliterations

#### Meta data files:

- *cunei_mzl.csv* contains the sign code class index established by Borger's Mesopotamisches Zeichenlexikon (MZL)
- *newLabels.json* contains new labels (re-indexing) for the subset of Neo-Assyrian MZL code classes so that labels range from 0-360 instead of 0-910 which reduces the output dimension of the detector
- *unicode_sign_stats.csv* contains estimates for sign length and height for individual cuneiform sign classes. These estimates were derived from the [Unicode Cuneiform Fonts](https://www.hethport.uni-wuerzburg.de/cuneifont/) by Sylvie Vanseveren. 


