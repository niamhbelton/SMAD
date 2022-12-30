# SMAD

This is the official repository for Siamese Network for Artifact Motion Detection (SMAD).

Please download the data from the following link;
https://openneuro.org/datasets/ds004173/versions/1.0.2

The code takes the following arguments;
```
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--N', type=int, default = 30)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', default = 1e-5, type=float)
    parser.add_argument('--mean_vec', type=int,default = 0)
    args = parser.parse_args()
    return args
```
- data_path is the path to where the data is stored
- model_name is to specify the model name
- output_name is the name of the files that are output
- N is the number of MRIs in the training data
- epochs is the number of epochs
- lr is the learning rate 
- mean vec is N/A, an experimental method for assigning scores to test images.


Example for training the model:
```
python3 train.py --data_path ./data/ds004173-download/ --model_name model --N 30 --epochs 20 --output_name results
```

Data is split into 30 images randomly selected medium to good quality MRIs (neuroradiolgist labels of one and two) and the remaining images are used for testing. 'metadata.csv' lists the names of the MRIs that were used for training.

The code creates an output directory and writes 
- training losses to file 'losses_<model_name>'
- the reference set features vector representations to '<output_name>_epoch_<epoch>_ref_vecs'
- the test set features vector representations to '<output_name>_epoch_<epoch>_test_vecs'
- the metrics below are written to '<output_name>_epoch_<epoch>_metrics'
  - specificity (based on MRIs labelled as one being 'good quality' and MRIs labelled as two and three as 'bad quality' and assigning test images a score based on the meaan distance to the reference set)
  - sensitivity (based on MRIs labelled as one being 'good quality' and MRIs labelled as two and three as 'bad quality' and assigning test images a score based on the mean distance to the reference set)
  - balanced accuracy (based on MRIs labelled as one being 'good quality' and MRIs labelled as two and three as 'bad quality' and assigning test images a score based on the mean distance to the reference set)
  - AUC (based on MRIs labelled as one being 'good quality' and MRIs labelled as two and three as 'bad quality' and assigning test images a score based on the mean distance to the reference set)
  - AUC (based on MRIs labelled as one being 'good quality' and MRIs labelled as two and three as 'bad quality' and assigning test images a score based on the minimum distance to the reference set)
    - AUC (based on MRIs labelled as one and two being 'good quality' and MRIs labelled as three as 'bad quality' and assigning test images a score based on the minimum distance to the reference set)
    - specificity (based on MRIs labelled as one and two being 'good quality' and MRIs labelled as three as 'bad quality' and assigning test images a score based on the minimum distance to the reference set)
    - sensitivity (based on MRIs labelled as one and two being 'good quality' and MRIs labelled as three as 'bad quality' and assigning test images a score based on the minimum distance to the reference set)
    - balanced accuracy (based on MRIs labelled as one and two being 'good quality' and MRIs labelled as three as 'bad quality' and assigning test images a score based on the minimum distance to the reference set)



