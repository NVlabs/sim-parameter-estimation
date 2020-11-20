# The Simulation Parameter Estimation Benchmark

The code accompaniment for the CoRL 2020 paper: [A User's Guide to Calibrating Robotics Simulators](https://corlconf.github.io/paper_286/#a-users-guide-to-calibrating-robotic-simulators) ([Arxiv](https://arxiv.org/abs/2011.08985)). This work was done at NVIDIA Research in Seattle.

**Abstract**: <p align="justify">Simulators are a critical component of modern robotics research. Strategies for both perception and decision making can be studied in simulation first before deployed to real world systems, saving on time and costs. Despite significant progress on the development of sim-to-real algorithms, the analysis of different methods is still conducted in an ad-hoc manner, without a consistent set of tests and metrics for comparison. This paper fills this gap and proposes a set of benchmarks and a framework for the study of various algorithms aimed to transfer models and policies learnt in simulation to the real world. We conduct experiments on a wide range of well known simulated environments to characterize and offer insights into the performance of different algorithms. Our analysis can be useful for practitioners working in this area and can help make informed choices about the behavior and main properties of sim-to-real algorithms. We open-source the benchmark, training data, and trained models, which can be found in this repository.</p>

This requires Python3.6, MuJoCo, and OpenAI Gym to run.

To run, use the following command:

```
python3.6 -m experiments.experiment_driver {NAME_OF_ESTIMATOR}
```

Our release is **in progress**, and you can track it below:

- [x] Estimators
- [ ] Trajectory Data
- [x] Plotting Code
- [x] MuJoCo Environments
- [ ] IsaacGym Environments

## Reference

```

@InProceedings{mehta2020calibrating,
  title={A User's Guide to Calibrating Robotics Simulators},
  author={Mehta, Bhairav and Handa, Ankur and Fox, Dieter and Ramos, Fabio},
  booktitle={Proceedings of the Conference on Robot Learning},
  year={2020},
  url={https://arxiv.org/abs/2011.08985},
}

```