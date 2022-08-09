Python porting of [Ito et al., "Low computational-cost cell detection method for calcium imaging data", Neuroscience Research, 2022](https://www.sciencedirect.com/science/article/pii/S016801022200075X)

# This repository is not ready.

### Quickstart

1. clone this repository

```bash
git clone https://github.com/magnetizedCell/lccd-python.git
```

2. install required packeages.

```bash
pip3 install -r lccd-python/requirements.txt
```

3.
   -  execute lccd in command line  
      ```bash
      cd lccd-python
      python3 lccd_python/lccd.py --conf config/config.json --src profiles/src.npy --dst out.dump
      ```
      Here source file is a numpy array file that have dimesion (X, Y, T).
   - use lccd as module  
      read profiles/test_lccd.ipynb

### TODO
- [x]Compare result with official implementation  
  Especially roi_integration -- seems ok
- []Make commnad line src file loading more flexible
- []Make variable and function names pythonic
- []Add GPU mode (see tests/profile_blob_detector.ipynb)
- []Add lazy frame loading for memory-efficient computation
- [x]Try sparse matrix in roi_integration: specify sparse:true in config.config.json
