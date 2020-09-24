# Multi-Agent Sensing

This package provides centralized implementations of Randomized Sequential
Partitions (RSP) (see references) that are suitable for numerical simulations as
well as a number of other algorithms for multi-agent planning via submodular
maximization such as sequential planning and auction algorithms.

Additionally, we implemented two application scenarios
* A simple coverage problem. Agent centers are distributed across the unit
  square. Actions are disks near the agent centers.
* A mutual information based target tracking problem. Agents plan on (short)
  receding horizons via Monte Carlo tree search.

## Dependencies

In addition to a number of registered Julia packages, some scripts for plotting
data from experimental trials rely on
[RosDataProcess](https://github.com/mcorah/RosDataProcess)
which is not yet registered with the Julia ecosystem.

## References

If you use this package in published work, pleace consider citing either of the
following:

For the coverage scenario and the initial implementation of RSP:
```
@inproceedings{corah2018cdc,
  author = {Corah, Micah and Michael, Nathan},
  title={Distributed Submodular Maximization on Partition Matroids for Planning
         on Large Sensor Networks},
  booktitle={Proc. of the {IEEE} Conf. on Decision and Control},
  year = {2018},
  month = dec,
  address = {Miami, FL},
}
```

Please cite the thesis for the target tracking scenario and everything else:
```
@phdthesis{corah2020phd,
  author = {Corah, Micah},
  title = {Sensor Planning for Large Numbers of Robots},
  school = {Carnegie Mellon University},
  year = {2020}
}
```
