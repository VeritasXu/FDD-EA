## Reliability 

1. For Windows, ensure the latest [Visual C++ runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) ([install link](https://aka.ms/vs/16/release/vc_redist.x64.exe)) is installed before using Ray
2. Update six before installing Ray
3. Ray

### Install sequence

- Visual C++ runtime, download and install (If you have installed visual studio 201x, this step can be ignored)

```tex
    https://aka.ms/vs/16/release/vc_redist.x64.exe
```

- Update six

```tex
    pip install --upgrade six
```

- Install ray

```tex
    pip install -U ray==1.3.0
```

- Install PyDOE

```tex
    pip install PyDOE
```

### Debug Mode

- IID: **FDD_EA.py**, adjust the hyper-parameters in config
- Non-IID: FDDEA_nIID.py, adjust the hyper-parameters in config

### Experimental Mode

- IID: run_FDD_EA.py, adjust the hyper-parameters in py file
- Non-IID: run_FDD_EA.py, adjust the hyper-parameters in py file

**Note**:

1. If there exists warning: FutureWarning: Not all Ray CLI dependencies were found. In Ray 1.4+, the Ray CLI.... , please run the following command in terminal:

```tex
    pip install ray[default]==1.3.0
```

2. If dashboard cannot start, you can disable it with 

```tex
    ray.init(include_dashboard=False)
```
instead of ray.init() in FDD-EA.py.

### Citation
If this code helps you in your research, please cite the following publication:

```tex
@article{xu2021federated,
  title={A federated data-driven evolutionary algorithm},
  author={Xu, Jinjin and Jin, Yaochu and Du, Wenli and Gu, Sai},
  journal={Knowledge-Based Systems},
  volume={233},
  pages={107532},
  year={2021},
  publisher={Elsevier}
}
```

And the multi-objective extension can be found at [FDD-MOEA](https://github.com/VeritasXu/FDD-MOEA.git) .
