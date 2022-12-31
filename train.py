import torch
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import argparse
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import f1_score

import numpy as np
import nibabel as nib


import torch.utils.data as data
import pandas as pd
import random


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1):
        x1 = torch.squeeze(x1, dim=0)
        features = self.pretrained_model.features(x1)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features1 = torch.max(pooled_features, 0, keepdim=True)[0]
        x1 = self.sigmoid(flattened_features1)
        return x1


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        return loss_contrastive


class Dataset(data.Dataset):
    def __init__(self, root_dir, N, task, transform=None):
        super().__init__()
        self.paths = []
        self.labels=[]
        scores = pd.read_csv(root_dir+ '/derivatives/scores.tsv', sep='\t')



        normals = scores.loc[scores['score'] == 1].index.values
        random.seed(0)
        samp = random.sample((list(normals)), N)


        if task == 'train':
            self.indexes = samp
        else:
            self.indexes = [x for i,x in enumerate(scores.index.values) if (i not in samp) ]

        for im in self.indexes:
            self.paths.append(root_dir+ '/' + scores.iloc[im,0].split('_') [0] + '/anat/' + scores.iloc[im,0] + '.nii.gz')
            self.labels.append(scores.iloc[im,1])

        self.root_dir = root_dir
        self.task = task



        self.transform = transform






    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        indexes = list(range(0,len(self.paths)))



        file_path = self.paths[index]

        img = nib.load(file_path)
        array = np.array(img.dataobj)



     #   array = array.reshape(array.shape[2], array.shape[0], array.shape[1])


        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        if self.task == 'train':
            ind = np.random.randint(len(indexes) + 1) -1
            while (ind == index):
                ind = np.random.randint(len(indexes) + 1) -1

            file_path =  self.paths[indexes[ind]]

            img = nib.load(file_path)
            array2 = np.array(img.dataobj)

       #     array2 = array2.reshape(array2.shape[2], array2.shape[0], array2.shape[1])


            if self.transform:
                array2 = self.transform(array2)
            else:
                array2 = np.stack((array2,)*3, axis=1)
                array2 = torch.FloatTensor(array2)

            label = torch.FloatTensor([0])
            label2=label
        else:
            array2=array

            if self.labels[index] == 1:
                label = torch.FloatTensor([0])
            else:
                label = torch.FloatTensor([1])

            if (self.labels[index] == 1) | (self.labels[index] == 2):
                label2 = torch.FloatTensor([0])
            else:
                label2 = torch.FloatTensor([1])






        return array, array2, label, label2

def evaluate(model, output_name, N, data_path, epoch, write_test=False):
  model.eval()

  ref_dataset = Dataset(data_path, N,'train')
  val_dataset = Dataset(data_path, N,'val')
  loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
  d={} #a dictionary of the reference images
  outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
  ref_images={} #dictionary for feature vectors of reference set
  #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
  if args.mean_vec == 0:
      model.to('cpu')
      for i in range(0, len(ref_dataset.indexes)):
          print(i)

          d['compare{}'.format(i)],_,_,_ = ref_dataset.__getitem__(i)
          ref_images['images{}'.format(i)] = model.forward( d['compare{}'.format(i)].unsqueeze(0))
          del d['compare{}'.format(i)]
          outs['outputs{}'.format(i)] =[]



      means = []
      mins = []
      lst=[]
      labels2=[]
      #loop through images in the dataloader
      for i,img in enumerate(loader):
          print('i in in loader {}'.format(i))
          image = img[0]

          lst.append(img[2].item())
          labels2.append(img[3].item())

          sum =0
          out = model.forward(image )
          minimum=100000
          for j in range(0,  len(ref_dataset.indexes)):

              euclidean_distance = F.pairwise_distance(out, ref_images['images{}'.format(j)])
              if euclidean_distance.detach().cpu().numpy()[0] < minimum:
                  minimum = euclidean_distance.detach().cpu().numpy()[0]
              outs['outputs{}'.format(j)].append(euclidean_distance.detach().cpu().numpy()[0])
              sum += euclidean_distance.detach().cpu().numpy()[0]
              del euclidean_distance

          means.append(sum /(j+1))
          mins.append(minimum)
          del image
          del out

          torch.cuda.empty_cache()


  elif args.mean_vec == 1:
         model.to('cpu')
         for i in range(0, 10):
            if i ==0:
                ref_im = ref_dataset.__getitem__(i)[0].unsqueeze(0)
                mean_vec = model.forward(ref_im)
                del ref_im

            else:
                ref_im = ref_dataset.__getitem__(i)[0].unsqueeze(0)
                mean_vec += model.forward(ref_im)
                del ref_im


         mean_vec = mean_vec / len(ref_dataset.indexes)
         means = []
         lst=[]
         labels=[]
         #loop through images in the dataloader
         for i,img in enumerate(loader):
             print('i in in loader {}'.format(i))
             image = img[0]

             lst.append(img[2].item())

             sum =0
             out = model.forward(image )

             print(out.shape)
             print(mean_vec.shape)
             euclidean_distance = F.pairwise_distance(out, mean_vec)
             print(euclidean_distance.shape)


             means.append(euclidean_distance.detach().cpu().numpy()[0])
             del image
             del out

             torch.cuda.empty_cache()



  df = pd.concat([pd.DataFrame(val_dataset.indexes), pd.DataFrame(lst), pd.DataFrame(labels2), pd.DataFrame(means), pd.DataFrame(mins)], axis =1)



  cols = ['id','label','label2','mean','min']


  df.columns=cols
  df = df.sort_values(by='mean', ascending = False).reset_index(drop=True)
  print(df)
  fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['mean']))
  auc = metrics.auc(fpr, tpr)
  fpr, tpr, thresholds = roc_curve(np.array(df['label2']),np.array(df['mean']))
  auc2 = metrics.auc(fpr, tpr)
  fpr, tpr, thresholds = roc_curve(np.array(df['label2']),np.array(df['min']))
  auc3 = metrics.auc(fpr, tpr)
  print('AUC is {}'.format(auc))
  print('AUC 2 is {}'.format(auc2))
  print('AUC 2 based on min is {}'.format(auc3))

  thres = get_threshold(ref_images)
  print(thres)
  thres = np.percentile(df['mean'], 49)
  print(thres)
  preds = np.where(df['mean'] > thres, 1, 0)
  df['preds'] = preds
  f1 = f1_score(np.array(df['label2']),np.array(df['preds']))
  fp = len(df.loc[(df['preds'] == 1 ) & (df['label2'] == 0)])
  tn = len(df.loc[(df['preds']== 0) & (df['label2'] == 0)])
  fn = len(df.loc[(df['preds'] == 0) & (df['label2'] == 1)])
  tp = len(df.loc[(df['preds'] == 1) & (df['label2'] == 1)])
  spec = tn / (fp + tn)
  sense = tp / (tp+fn)
  recall = tp / (tp+fn)
  acc = (sense + spec) / 2

  print('Specificity is {}.'.format(spec))
  print('Sensitivity is {}.'.format(sense))
  print('Balanced Accuracy is {}.'.format(acc))


  thres = np.percentile(df['min'], 49)
  print(thres)
  preds = np.where(df['min'] > thres, 1, 0)
  df['preds'] = preds
  f1 = f1_score(np.array(df['label2']),np.array(df['preds']))
  fp = len(df.loc[(df['preds'] == 1 ) & (df['label2'] == 0)])
  tn = len(df.loc[(df['preds']== 0) & (df['label2'] == 0)])
  fn = len(df.loc[(df['preds'] == 0) & (df['label2'] == 1)])
  tp = len(df.loc[(df['preds'] == 1) & (df['label2'] == 1)])
  spec2 = tn / (fp + tn)
  sense2 = tp / (tp+fn)
  recall2 = tp / (tp+fn)
  acc2 = (sense2 + spec2) / 2



  #write out the reference images
  ref = pd.DataFrame(ref_images['images{}'.format(0)].detach().numpy())
  for i in range(len(ref_images)):
      ref = pd.concat([ref,  pd.DataFrame(ref_images['images{}'.format(i)].detach().numpy())], axis =0)

  vec_name = output_name + '_ref_vecs_epoch_' + str(epoch)
  ref.to_csv('./outputs/' +vec_name)



  if write_test == True:
        outs = []
        for i,img in enumerate(loader):
            image = img[0]
            outs.append(model.forward(image ))

        test = pd.DataFrame(outs[0].detach())
        for i in range(len(outs)):
            test = pd.concat([test,  pd.DataFrame(outs[i].detach().numpy())], axis =1)

        out_name = output_name + '_test_vecs_epoch_' + str(epoch)
        test.to_csv('./outputs/' +out_name)


  name = output_name + '_metrics_epoch_' + str(epoch)
  pd.DataFrame([spec, sense, acc, auc, auc2, auc3, spec2, sense2, acc2]).to_csv('./outputs/' + name)

  df.to_csv('./outputs/' +output_name)
  print('Writing output to {}'.format(('./outputs/' +output_name)))


def get_threshold(ref_images):
    dist = 0
    for i in range(0, len(ref_images)):
        for j in range(i, len(ref_images)):
            if i != j:
                dist += F.pairwise_distance(ref_images['images{}'.format(i)].detach(), ref_images['images{}'.format(j)].detach())


    dist =  dist / (len(ref_images) -1)
    return dist.numpy()[0]



def train(model, train_dataset, epochs, criterion, model_name, indexes):
    device='cuda'
    avg_losses=[]

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    for epoch in range(epochs):
        model.to(device)
        model.train()
        print("Starting epoch " + str(epoch+1))
        ind = list(range(len(indexes)))
        loss_sum = 0
        for index in ind:
            img1, img2, labels,_ = train_dataset.__getitem__(index)
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            output1 = model.forward(img1)
            output2 = model.forward(img2)
            loss = criterion(output1,output2,labels)
            loss_sum += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del img1
            del img2
            del labels


        avg_losses.append(loss_sum / len(ind))
        pd.DataFrame(avg_losses).to_csv('./outputs/losses_' + model_name)
        torch.save(model, './outputs/' + model_name)

    if evaluate_epoch != 1:
            out_name = args.output_name + '_epoch_' + str(epoch)
            evaluate(model, out_name, args.N, args.data_path, epoch,True)

    pd.DataFrame(avg_losses).to_csv('./outputs/losses_' + model_name)
    print("Finished Training")


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


if __name__ == '__main__':
    args = parse_arguments()
    train_dataset = Dataset(args.data_path, args.N,'train')
    #set the seed
    weight_init_seed=1001
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)
    model = Net()
    epochs=args.epochs
    model_name = args.model_name
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = ContrastiveLoss()
    train(model, train_dataset, args.epochs, criterion, args.model_name, train_dataset.indexes)
  #  evaluate(model, args.output_name, args.N, args.data_path, epoch, True)
