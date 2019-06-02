# Implementation in TF 2.0.0a

Paper implementation of MST via graph cuts.


## Results
To be added later.

## Training

### Data Structure

To load the data we assume the following structure, both locally and remotely:
```
trainer
-- data
  -- content
    -- content_images_*.png
  -- style
    -- style_images_*.png
```

### Local
First, create the following project structure locally.
```
git clone repo
cd repo/trainer/
mkdir data
cd data/
mkdir content
mkdir style
```

Then simply run the shell script to start training

```
sh train.sh local
```

### Remote - GCP
Edit `train.sh` to specify your `datapath=gs://path_to_data_dir/` variable. 
```
sh train.sh remote
```

## Transfer Learning
If you wish to use pretrained decoder weights, specify the path to them using `weights` parameter in the `train.sh` file.




