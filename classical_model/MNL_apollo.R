# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "MNL_2762023",
  modelDescr      = "MNL model with incentive data",
  indivID         = "user_id",
  #mixing          = TRUE,
  #nCores          = 5, 
  #workInLogs      =TRUE,
  outputDirectory = "D:/Machine learning project/PP_incentivesurvey/PP/ICLV model/Results"
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
### if data is to be loaded from a file (e.g. called data.csv), 
### the code would be: database = read.csv("data.csv",header=TRUE)

##### DATA PREPARATION #### 

database = read.csv("D:/Machine learning project/PP_incentivesurvey/PP/ICLV model/data_hybrid_1.csv")

database$walkavail<- ifelse(database$mode=="walk",1, ifelse(database$cardistance<=7,1,0))

database$information2<- ifelse(database$information=="no info",0,1)

database$female <- ifelse(database$SEX==2,1,0)
database$inc1 = ifelse(database$INCOME==1,1,0)
database$inc2 = ifelse(database$INCOME==2,1,0)
database$inc3 = ifelse(database$INCOME==2,1,0)
database$inc4 = ifelse(database$INCOME==4,1,0)
database$linc <- ifelse(database$INCOME<3, 1, 0)
database$yage <- ifelse(database$AGE < 31, 1, 0)
database$bicycleincentive <- ifelse(database$incentivezone==1,database$bicycleincentive,0)
database$busincentive <- ifelse(database$incentivezone==1,database$busincentive,0)
database$carincentive <- ifelse(database$incentivezone==1,database$carincentive,0)
database$motorincentive <- ifelse(database$incentivezone==1,database$motorincentive,0)
database$trainincentive <- ifelse(database$incentivezone==1,database$trainincentive,0)
database$walkincentive <- ifelse(database$incentivezone==1,database$walkincentive,0)


#### dummies for purpose ####
database$commute = ifelse(database$Purpose ==100,1,0)
database$return = ifelse(database$Purpose ==101,1,0)
database$shopD = ifelse(database$Purpose ==200,1,0)
database$shopND = ifelse(database$Purpose ==201,1,0)
database$leisure = ifelse(database$Purpose ==202|database$Purpose==600,1,0)
database$business = ifelse(database$Purpose ==300,1,0)
database$hospital = ifelse(database$Purpose ==400,1,0)
database$fixed = ifelse(database$Purpose ==100|database$Purpose==300|database$Purpose==400,1,0)
#### DUMMIES FOR JOB TYPE ##### 

database$civil = ifelse(database$job_type=="civil servant",1,0)
database$company = ifelse(database$job_type=="company employee",1,0)
database$housewife = ifelse(database$job_type=="Housewife",1,0)
database$mangex = ifelse(database$job_type=="Management executive",1,0)
database$parttime = ifelse(database$job_type=="Part time job",1,0)
database$freelance = ifelse(database$job_type=="Self employed/ Freelance",1,0)
database$fulltime = ifelse(database$job_type=="civil servant"|database$job_type=="company employee"|database$job_type=="Management executive",1,0)
#database$fulltime = ifelse(database$job_type=="company employee",1,0)
### DUMMIES for INFORMATION ### 
database$envinfo = ifelse(database$recco==3|database$recco==2,1,0)
database$heainfo = ifelse(database$recco==3|database$recco==4,1,0)
database$noinfo = ifelse(database$recco==1,1,0)
### Log of income ## 
database$loginc = log(database$INCOME)

### Preparing incentive and penalty values for box-cox transformations 

database$incen1 = database$bicycleincentive + 1
database$incen2 = database$busincentive + 1
database$incen3 = -1*database$carincentive + 1
database$incen4 = -1*database$motorincentive + 1
database$incen5 = database$trainincentive + 1
database$incen6 = database$walkincentive + 1

##### Dummies for information 

## Information while they are receiving incentive 
database$info_inc_bike = ifelse(database$bicycleincentive> 0 & database$noinfo==0,1,0)
database$info_inc_bus = ifelse(database$busincentive >0 & database$noinfo ==0,1,0)
database$info_inc_car = ifelse(database$carincentive < 0 & database$noinfo ==0,1,0)
database$info_inc_motor = ifelse(database$motorincentive < 0 & database$noinfo ==0,1,0)
database$info_inc_pub = ifelse(database$trainincentive > 0 & database$noinfo ==0,1,0)
database$info_inc_walk = ifelse(database$walkincentive > 0 & database$noinfo ==0,1,0)

## Information without incentive
database$info_ninc_bike = ifelse(database$bicycleincentive==0 & database$noinfo ==0,1,0)
database$info_ninc_bus = ifelse(database$busincentive ==0 & database$noinfo ==0,1,0)
database$info_ninc_car = ifelse(database$carincentive ==0 & database$noinfo ==0,1,0)
database$info_ninc_motor = ifelse(database$motorincentive == 0 & database$noinfo ==0,1,0)
database$info_ninc_pub = ifelse(database$trainincentive == 0 & database$noinfo ==0,1,0)
database$info_ninc_walk = ifelse(database$walkincentive == 0 & database$noinfo ==0,1,0)

database$info_inc = ifelse(database$incentivezone==1 & database$noinfo ==0, 1, 0)
# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(asc_1 =  0, asc_2  =  0,asc_3  =  0,asc_4   = 0,asc_5  =  0,
                b_tc           = 0, 
                b_tt            = 0 , 
                b_inc=0) 
                
                



## Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### Likelihood of choices
  
  V = list()
  
  V[['bike']] = asc_1 + b_tt*biketime+ b_inc*bicycleincentive 
   
  
  
  V[['bus']] = asc_2 + b_tc*buscost+ b_tt*bustime + b_inc*busincentive 
  
  
  V[['car']] = b_tc*carcost+b_tt*cartime + b_inc*carincentive 
  
  V[['motor']] = asc_3+b_tc*motorcost+b_tt*motortime + b_inc*motorincentive 
  
  V[['pub']] = asc_4 + b_tc*pubcost+ b_tt*pubtime+ b_inc*trainincentive
   
  V[['walk']] =  asc_5 + b_tt*walktime+ b_inc*walkincentive 
  
  
  
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(bike='bike', bus='bus', car='car', motor='motor', pub='pub', walk='walk'),#, taxi='taxi'
    avail         = list(bike=bikeavail, bus=busavail, car=1, motor=motoravail, pub=pubavail, walk=walkavail), #,taxi=1
    choiceVar     = mode,
    V             = V
  )
  
  ### Compute probabilities using MNL model
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}
# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #

apollo_saveOutput(model)

