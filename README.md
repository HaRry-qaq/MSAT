# MSAT

## Demo preview

Please visit

```sh
HaRry-qaq.github.io
```
to listen to different kinds of experiment demos.

## Set up development environment

Please go to MSAT/msat. You can create the environment with the following command.


```sh
conda env create -f environment.yml
```


## Experiment I: MMT-note MMT-note and MMT-track.

Please go to baseline/mmt train the MMT-note, MMT-bar, MMT-track.  

## Preprocessing

The relevant code for preprocessing is in baseline/mmt. Go to that folder under.

You can get the SOD dataset by 
```sh
wget https://qsdfo.github.io/LOP/database/SOD.zip.
```


### Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name *.mid -o -name *.xml | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

### Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python mmt/convert_sod.py
```

### Extract the note-level representation

Extract a list of notes from the MusPy JSON files.

```sh
python mmt/extract.py -d sod
```

### Cut the seq_len to 1024
Enter the file address in the corresponding place of the code.

```sh
python mmt/cuthang_1024.py
```

### Extract the bar-level representation/track-level representation

sort the note-level representation to bar-level and track-level representation, Remember to modify the specified path within the file.

```sh
python mmt/representation-bar.py -d sod
python mmt/representation-track.py -d sod
```

### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python mmt/split.py -d sod
```

## Prepare the data set

Create the corresponding sod and sod/processed/note folder in the data folder. Put the csv obtained by cuthang_1024.py in sod/processed/note and the txt obtained by split.py in sod/processed.

Create the corresponding sod-bar and sod-bar/processed/note folder in the data folder. Put the csv obtained by representation-bar.py in sod-bar/processed/note and the txt obtained by split.py in sod/processed.

Create the corresponding sod-track and sod-track/processed/note folder in the data folder. Put the csv obtained by representation-track.py in sod-track/processed/note and the txt obtained by split.py in sod/processed.

## Training MMT-note MMT-note and MMT-track

Please go to baseline, train the MMT-note, MMT-bar, MMT-track.  

MMT-note:

```sh
 python mmt/train.py -d sod -o exp/sod/ape -g 0
```

MMT-bar:

```sh
 python mmt/train.py -d sod-bar -o exp/sod-bar/ape -g 0
```

MMT-track:

```sh
 python mmt/train.py -d sod-track -o exp/sod-track/ape -g 0
```


## Experiment II: MSAT-LA and MSAT-GA

Go to the corresponding folder MSAT/.

Load the previously trained MMT-note and MMT-track into the model, i.e., change the path1 and path2 specified in the train_multi_scale.py.

Prepare the sod-bar data under MSAT/ in the same way as MMT-bar.

Run the following code to train the remaining parameters:

  `python -m torch.distributed.launch --nproc_per_node=3 msat/train_multi_scale.py -o exp/sod-bar/ape`

  
## Generation
Generate new samples using a trained model.

```sh
python msat/generate.py -d sod-bar -o exp/sod-bar/ape -g 0
```

## Evaluation
Evaluate the trained model.

Go to MSAT/evaluate/.

Modify the path specified in the py file and run the following code:
```sh
python evaluate-local.py
```
```sh
python evaluate-instrument-corr.py
```
```sh
python evaluate-track-var.py
```
