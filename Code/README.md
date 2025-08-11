# Elsevier 2025 journal paper - MmWave beam path blockage prevention through codebook value prediction under domain shift
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

### Requirements
In the requirements.txt file the needed python modules for this project are specified.

### Preprocessing steps
The pre-processed dataset in '../Data' is created using the pre-processing scripts in 'pre-processing/'.
Links to the original dataset have been added to this subdirectory as well in 'pre-processing/Radar data/Radar1' and 'pre-processing/Radar data/Radar2'.
The pre-processing scripts are not part of the audit procedure and can therefore not be run automatically.

Steps required in order to use the pre-processing scripts:

1) Download entire original dataset to a directory (more instruction is provided in the respective Radar subfolders).
2) Download '../Code' subdirectory to a directory.
3) Using python 3.7 < ver. < 3.9, install imported packages listed in the pre-processing script.
4) Place the .bin files of Radar1 and Radar2 in their respective 'Radar data' subfolders (more instruction is provided in the respective subfolders).

Follow a set of steps:

### bin_reading.py

5) In the pre-processing script, update the top directory prefix of 'pre-processing\Radar' to the absolute location inside the file system. The os.walk() function will list files recursively.
6) Linux users: the rsplit(.) function calls and directory string depend on quoted Windows separators. Adjust the separators to Linux separators accordingly.
7) Execute pre-processing script.

### Reproducing results

The run.sh file includes all bash commands which should be run to acquire the results used in Figures 7-10 of the journal paper.
Prior to running the bash commands, make sure that all packages listed in the requirements.txt file are installed.
Prior to running the bash commands, copy the datasets inside the '../Data' directory to the 'Datasets/' directory.
The results in .csv format are placed in subdirectories per experiment type inside the results/ directory.

Bash command for acquiring results presented in Figure 6 is not provided. Reason being that this figure is considered to be part of an offline pre-processing procedure. When there is a desire to acquire results that have been used inside the figure, run fresnelRadiusCalculation.py and plotPathLossTime.py in succint fashion located in 'pre-processing/worst-case Fresnel ellipsoid radius simulation'. Please adjust the file paths if needed accordingly (paths have been set on a Windows system).

The Java Processing label simulation sketch files used for generating task labels are included in 'pre-processing/curvePoint label simulation'.

### Plots

Figure plots inside the journal paper were created by processing the .csv formatted results with MS Excel into a chart structure.

### Leave-out

In every .csv file, the last 4 columns denote the summary statistics achieved on a specifically held out test dataset.
The data inside these columns for different cross validation splits should be grouped per machine learning technique and domain factor held out type.
On the grouped data, AVERAGE and VAR.S functions should be called per metric.
The function outputs should be structured according to a bar chart structure comparable to the structure used inside the journal paper for Figure 9.

### Frame-ablation

In every .csv file, columns denote the summary statistics achieved on a specifically held out test dataset. The rows denote prediction window size.
The data inside these columns for different machine learning techniques should be grouped per domain factor held out type.
The grouped data should be plotted as separate line graphs in separate figures per domain factor held out type.

### Shapley

The data in the .csv files can directly be added to a bar chart separately per machine learning technique.
The header numbers denote dimension index in the mini-batch tensor and superpixel index in the respective dimension (eg. when there are 256 fast-time indices, 1 refers to the second group of 64 fast-time indices).

### Remove-retrain

In every .csv file, columns denote the summary performance statistics achieved on a specifically held out test dataset. The rows denote succinctly zero-masked superpixel index.
The direct statistics should be plotted as separate line graphs in separate figures per machine learning technique.
The top-1 minimum validation subset loss statistic should be retrieved from the respective stdout file per machine learning technique manually in Streams/.
Note: make sure that run.sh does not delete Streams/ content to be able to do this!
The top-1 minimum validation subset loss statistic can be plotted as separate line graph in separate figures per machine learning technique.

