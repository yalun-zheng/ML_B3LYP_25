<img src="https://github.com/user-attachments/assets/25798da7-8e92-4778-8f27-f2e5945cefe9" alt="img" width="600" height="450">  

#### Step 1
Clone this repository into your device. Run  
```
git clone https://github.com/yalun-zheng/ML_B3LYP_25.git
```  
or
```
git clone git@github.com:yalun-zheng/ML_B3LYP_25.git
```      
Assume that the repository is cloned in /path_to_code/ML_B3LYP_25, then assign this path in cfg.yaml and in run.sh, separately, as follows.   
In cfg.yaml, change the first line homepath: PATH_TO_CODE with  
```
homepath: /path_to_code/ML_B3LYP_25
```     
Also in run.sh, change the first line cd PATH_TO_CODE with  
```
cd /path_to_code/ML_B3LYP_25
```

#### Step 2
Install some necessary packages.  
```numpy==1.26.0  
pyscf==2.3.0
opt_einsum
yaml
ml_collections
torch  
matplotlib
pandas
```
Make sure directories named with subsets name like `/path_to_code/ML_B3LYP_25/data/pkl/G2/`, `/path_to_code/ML_B3LYP_25/data/pkl/test/` or `/path_to_code/ML_B3LYP_25/data/pkl/other_dataset_name/`  are made.
#### Step 3
Run the script run.sh to train, valid or test a model.  
```
sh run.sh
```
The validation and test results using the default model will be saved in valid.csv and test.csv    

#### Note  
By editing cfg.yaml and giving the corresponding .xyz files, you can validate/test any structure using the default model I uploaded or any other model you trained.
