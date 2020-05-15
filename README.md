# Results from the paper "Asymptotic Reduction of a Lithium-ion Pouch Cell Model"

A pre-print of this manuscript is available [here](https://arxiv.org/abs/2005.05127).

The scripts use the open-source battery modelling software [PyBaMM](https://github.com/pybamm-team/PyBaMM). To install the appropriate version of PyBaMM, run:
```
pip install -e git+https://github.com/pybamm-team/pybamm.git@asymptotic-pouch-cell#egg=pybamm
```

The scripts used to generate the figures and tables are as follows:

- Figures 2-5: `compare_models`
- Figure 6: `make_error_plot`
- Table 1: `make_error_table_1plus1D` for the "1+1D" results and ` make_error_table_CC` for the "CC" results
- Figures SM1: `supplementary/plot_discharge_curve`
- Figures SM2-SM5: `supplementary/plot_potentials_concentrations`
- Table SM2: `supplementary/make_error_table`
