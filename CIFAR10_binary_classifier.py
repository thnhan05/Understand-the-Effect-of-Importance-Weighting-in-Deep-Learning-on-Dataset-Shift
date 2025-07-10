import os,logging,sys,time
import argparse
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
from torch.utils import data
import functools


import torchvision

import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from MIL import resnet_pytorch
from utilities import logger as log_module
from utilities import common_functions
import h5py
import joblib
import csv
import random

class ResNetClassifier(nn.Module):
    def __init__(self, resnet_type = 'resnet18', output_classes=1, fc_layer = True,random_init=False):
        super(ResNetClassifier, self).__init__()
        
        resnet_structure_dict = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3],
                                    "resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3]}
        resnet_block_dict = {"resnet18": BasicBlock, "resnet34": BasicBlock,
                                "resnet50": Bottleneck, "resnet101": Bottleneck}
        
        self.base_net = ResNetV1(64, resnet_block_dict[resnet_type], resnet_structure_dict[resnet_type], 3, False)
        if not random_init:
            pretrained_params = model_zoo.load_url(model_urls[resnet_type])
            def dn_preload_translate_logic(name):
                return "{0}{1}".format("", name)
            load_pretrained_nework(self.base_net,param_dict=pretrained_params,
                                name_translate_func=dn_preload_translate_logic)        
 
        self.fc_layer = fc_layer
        if self.fc_layer: 
            self.fc = nn.Linear(512, output_classes)
            self.sigmoid = nn.Sigmoid()
            

    def forward(self, x):
        x = self.base_net(x).view(-1,512)
        if self.fc_layer: 
            x = self.fc(x)
            x = self.sigmoid(x)
        return x


# Define the simpleCNN based on the description in the paper
class simpleCNN(nn.Module):
    def __init__(self, output_classes=2, dropout_rate = -999):
        super(simpleCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.net = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Linear(2048,512)
        if self.dropout_rate >= 0:
            self.dropout1 = nn.Dropout(p = self.dropout_rate)
            self.dropout2 = nn.Dropout(p = self.dropout_rate)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,output_classes)
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        if self.dropout_rate >= 0:
            x= self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.dropout_rate >= 0:
            x= self.dropout2(x)
        x = self.fc3(x)

        return x


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:     
            x = self.map(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  
        y = self.dataset[index][1]         
        return x, y

    def __len__(self):
        return len(self.dataset)


def process_CIFAR10(batch_size = 16, num_workers=2,weight=1):
    # Notice that we apply the same mean and std normalization calculated on train, to both the train and test datasets.

    transform_train = transforms.Compose([
                                        # transforms.RandomCrop(32, padding=4),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.4914, 0.4822, 0.4465], 
                                            [0.247, 0.243, 0.261])
                                        ])
    
    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.4914, 0.4822, 0.4465], 
                                            [0.247, 0.243, 0.261])
                                        ])

    trainset = CIFAR10(root='./cifar10', train=True, transform=None, target_transform=None, download=True)
    testset = CIFAR10(root='./cifar10', train=False, transform=None, target_transform=None, download=True)

    
    

    classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    def get_class_i(x, y, i):
        """
        x: trainset.train_data or testset.test_data
        y: trainset.train_labels or testset.test_labels
        i: class label, a number between 0 to 9
        return: x_i
        """
        # Convert to a numpy array
        y = np.array(y)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(y == i)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:,0])
        # Collect all data that match the desired label
        x_i = [x[j] for j in pos_i]

        return x_i
    
    x_train  = trainset.data
    y_train  = trainset.targets
    if weight <= 1:
        num_cats = int(5000*weight)
        num_dogs = 5000
    else:
        num_cats = 5000
        num_dogs = int(5000*(1/weight))
    
    cat_dog_data = np.array(get_class_i(x_train, y_train, classDict['cat'])+ get_class_i(x_train, y_train, classDict['dog']))
    cat_dog_labels = [0 for j in range(num_cats)] + [1 for j in range(num_dogs)]
    cat_dog_trainset = list(zip(cat_dog_data, cat_dog_labels))
    

    random_data = np.random.rand(1000,32,32,3).astype('float32')
    random_labels = [10 for j in range(1000)]
    random_test = list(zip(random_data, random_labels))

    trainset = MapDataset(cat_dog_trainset, transform_train)
    testset = MapDataset(testset, transform_test)
    randomset = MapDataset(random_test, transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    random_loader = DataLoader(randomset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader,test_loader, random_loader



class experiment():
    def __init__(self, model, input_channel,importance_weights,lr,device,learning_alg='sgd',weight_decay = 0):
        self.model = model.to(device)         
        self.criterion = torch.nn.CrossEntropyLoss(weight = importance_weights)
        self.input_channel = input_channel
        if learning_alg == 'sgd' : 
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=0.9, weight_decay= weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        

    def train(self,train_loader,device,save_path = None):
        y_labels=[]
        preds=[]
        losses = []
        metrics = dict()

        for step, (h, y) in enumerate(train_loader):
            h = h.float().to(device)
                
            y = y.long().to(device)

            outputs = self.model(h)
            
            loss = self.criterion(outputs, y)
            score, pred = torch.max(outputs, 1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate accuracy and save metrics
            losses.append(loss.item())
            y_labels.extend(y.data.cpu().numpy().tolist())
            preds.extend(pred.data.cpu().numpy().tolist())
        #print(outputs[0:10],y[0:10])
        preds = np.array(preds)
        y_labels = np.array(y_labels)

        acc = (preds == y_labels).mean()
        metrics["Loss/train"] = np.mean(losses)
        metrics["Acc/train"]= acc

        if save_path is not None:

            torch.save(self.model.state_dict(),save_path)

        return metrics


    def evaluate(self,data_loader,divide,device):
        y_labels=[]
        preds=[]
        losses = []
        metrics = dict()

        self.model.eval()      
        with torch.no_grad():
            for step, (h, y) in enumerate(data_loader):
                h = h.float().to(device)                
                y = y.long().to(device)

                outputs = self.model(h)
                
                score, pred = torch.max(outputs, 1)
                y_labels.extend(y.data.cpu().numpy().tolist())
                preds.extend(pred.data.cpu().numpy().tolist())

   
        preds = np.array(preds)

        y_labels = np.array(y_labels)
        """
        prediction = [3 if pred==0 else 5 for pred in preds]
        y_labels = [0 if y_label not in [3,5,11] else y_label for y_label in y_labels]
        """
        acc = (np.where(preds==0,3.0,5.0) == y_labels).mean() ### preds so far only pick 0/1
        metrics["Acc/test"] = acc
       

        cat_dog_labels = np.concatenate((preds[y_labels == 3.0],preds[y_labels == 5.0]),axis=0)
        
        other_labels = preds[y_labels != 3.0]
        other_labels = other_labels[y_labels[y_labels != 3.0]!=5.0]
   
        metrics['frac in all'] = np.mean(preds)
        if divide == True:
          metrics['frac in cat/dog'] = np.mean(cat_dog_labels)
          metrics['frac in others'] = np.mean(other_labels)

        return metrics




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/gpfs/data/geraslab/Yanqi/saves/toolsforML", type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    parser.add_argument("--model_name", default="simpleCNN",type=str, help="pre-trained model name (e.g. model-10.pt)")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for frozen representation.")
    parser.add_argument("--num_epochs", default=500, type=int, help="Number of epochs to train for.")
    parser.add_argument("--checkpoint_epochs",default=100,type=int,help="Number of epochs between checkpoints/summaries.",)
    parser.add_argument("--num_workers",default=10,type=int,help="Number of data loading workers (caution with nodes!)",)
    parser.add_argument("--learning_alg", default="sgd",type=str)
    parser.add_argument("--weight_decay", default=0,type=float)
    parser.add_argument("--seed", default=0,type=int)
    parser.add_argument("--dropout", default=-999.0,type=float)
    parser.add_argument("--unbalanced_weight", default= 1.0,type=float)
    parser.add_argument("--importance_weight",default="[1.0,1.0]",type=str)
    parser.add_argument("--random_init", action="store_true", default=False)

    args = parser.parse_args()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    model_saves_directory = os.path.join(args.model_path, args.model_name)
    assert not os.path.exists(model_saves_directory), "This model directory already exists"
    os.makedirs(model_saves_directory)

    
    if args.model_path is not None:
        file_name = os.path.join(model_saves_directory,"log")
        logging.basicConfig(filename= file_name, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("model_dir = {0}".format(model_saves_directory))
    logging.info("Parameters: {}".format(args))
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader,test_loader,random_loader = process_CIFAR10(batch_size =args.batch_size,num_workers=args.num_workers,weight=args.unbalanced_weight)

    model = simpleCNN(output_classes=2,dropout_rate = args.dropout)
    weights = torch.tensor(eval(args.importance_weight)).to(device)
    run_experiment = experiment(model=model, 
                                input_channel=3,
                                importance_weights=weights, 
                                learning_alg=args.learning_alg,
                                lr=args.learning_rate,
                                weight_decay = args.weight_decay,
                                device=device)   

    # add csv file for performance
    global_document_columns= ["epoch","phase","loss","acc","frac_in_all","frac_in_cat/dog","frac_in_others","frac_in_random"]
    global_document_unit = common_functions.DocumentUnit(global_document_columns)

    if model_saves_directory is not None:
        global_document_unit_save_dir = os.path.join(model_saves_directory, 'performance.csv')


    num_epochs = args.num_epochs
    # Train fine-tuned model
    for epoch in range(num_epochs):
        epoch += 1
        logging.info('Epoch number: {0} starts'.format(epoch))

        model_save_dir = os.path.join(model_saves_directory,'epoch_{0}.ckpt'.format(epoch)) if epoch % args.checkpoint_epochs == 0 else None 
        metrics = run_experiment.train(train_loader,save_path = model_save_dir,device=device)
        logging.info(f"Training Epoch [{epoch}/{num_epochs}]: " + "\t".join([f"{k}: {np.array(v)}" for k, v in metrics.items()]))
        global_document_unit.add_values("phase", ['training'])
        global_document_unit.add_values("epoch", [epoch])
        global_document_unit.add_values("loss", [metrics['Loss/train']])
        global_document_unit.add_values("acc", [metrics['Acc/train']])
        for item in ["frac_in_all","frac_in_cat/dog","frac_in_others","frac_in_random"]:
            global_document_unit.add_values(item,[-999])


        # Evaluate fine-tuned model
        test_metrics = run_experiment.evaluate(test_loader,divide=True,device=device)  
        random_metrics = run_experiment.evaluate(random_loader,divide=False,device=device)
        logging.info(f"Testing Epoch [{epoch}/{num_epochs}]: " + "\t".join([f"{k}: {np.array(v)}" for k, v in test_metrics.items()]))  
        logging.info(f"Testing(random) Epoch [{epoch}/{num_epochs}]: " + "\t".join([f"{k}: {np.array(v)}" for k, v in random_metrics.items()]))  

        global_document_unit.add_values("phase", ['testing'])
        global_document_unit.add_values("epoch", [epoch])
        global_document_unit.add_values("loss", [-999])
        global_document_unit.add_values("acc", [test_metrics['Acc/test']])
        global_document_unit.add_values("frac_in_all", [test_metrics['frac in all']])
        global_document_unit.add_values("frac_in_cat/dog", [test_metrics['frac in cat/dog']])
        global_document_unit.add_values("frac_in_others", [test_metrics['frac in others']])
        global_document_unit.add_values("frac_in_random", [random_metrics['frac in all']])

        global_document_unit.to_csv(global_document_unit_save_dir)
