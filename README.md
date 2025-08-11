# Elsevier Internet of Things journal paper - MmWave beam path blockage prevention through codebook value prediction under domain shift
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

### Use of the docker container
Note: steps listed below only work on a host machine running with a Linux OS.

1) Download the entire repository to a machine.

Call the following docker commands inside the repository:

2) docker build -t bberlo/elsevier-audit:2025.5 .
3) docker run --name bberlo_elsevier_audit -v $PWD/Data/:/project/Code/Datasets/ bberlo/elsevier-audit:2025.5 /bin/bash /project/Code/run.sh
4) (inside a new terminal) docker cp bberlo_elsevier_audit:/project/Code/results/ ./Code/results

Note: due to required storage capacity, the $PWD/Data/ directory is unavailable at GitHub. Please contact TU/e IRIS (https://iris.win.tue.nl/) for access.
Note: wait for a docker command to be finished running before executing the next docker command.
Note: In order to check if the docker run command is finished, inspect the terminal information of the run.sh command (docker run is not executed in detached mode).

### Plots

Please check the readme file inside the 'Code' directory on how the figure plots inside the journal paper should be created using the data inside the 'Code/results' directory.
