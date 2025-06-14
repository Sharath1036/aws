# AWS CLI Setup and Environment Configuration

## Install AWS CLI

1. Download and install the AWS CLI from [AWS CLI official website](https://aws.amazon.com/cli/)
2. Verify installation:
```
aws --version
```

## IAM User Creation
1. Go to AWS IAM Console
2. Navigate to Users > Create user
3. Enter username
4. For permissions:
* Select "Attach policies directly"
* Choose "AdministratorAccess" (or appropriate level of access)
5. Click Next > Create user

## Access Key Creation
1. Select the newly created user
2. Go to Security credentials tab
3. Under Access keys, click Create access key
4. Select CLI use case
5. Click Next > Create access key
6. Download the CSV file containing:
* Access key ID
* Secret access key

## Configure AWS CLI
```
aws configure
```
When prompted, enter:
1. AWS Access Key ID (from downloaded CSV)
2. AWS Secret Access Key (from downloaded CSV)
3. Default region name (e.g., us-east-1)
4. Default output format (e.g., json)

## Project Setup
### Required Files
```
project/
├── requirements.txt
├── dataset.csv
├── notebook.ipynb
```

### Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate     # Windows
```

### Install Dependencies
```
pip install -r requirements.txt
```

Typical requirements might include:
```
text
sagemaker>=2.0
ipykernel
pandas
numpy
scikit-learn
boto3
```

## Create S3 Bucket
1. Go to Amazon S3 Console
2. Click Create bucket
3. Enter unique bucket name
4. Select region
5. Leave other settings as default (or configure as needed)
6. Click Create bucket

Verification
```
aws s3 ls
```

## Setup and Data Loading

```python
import sagemaker
from sklearn.model_selection import train_test_split
import boto3
import pandas as pd

sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'mobbucketsagemaker' # Mention the created S3 bucket name here
print("Using bucket " + bucket)
```
```
df = pd.read_csv("mob_price_classification_train.csv")
```
## Data Exploration
### Display First Rows
```
df.head()
df.shape  # Output: (2000, 21)
```

### Target Variable Distribution
```
df['price_range'].value_counts(normalize=True)
```
Output shows equal distribution across 4 price ranges (0.25 each).

### Column Names
```
df.columns
```
### Missing Values Check
```
df.isnull().mean() * 100
```
No missing values found in any column.

## Data Preparation
### Feature and Label Separation
```
features = list(df.columns)
label = features.pop(-1)  # 'price_range' is the target
```
```
x = df[features]
y = df[label]
```

## Train-Test Split
```
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
print(X_train.shape)  # (1700, 20)
print(X_test.shape)   # (300, 20)
print(y_train.shape)  # (1700,)
print(y_test.shape)   # (300,)
```

### Save Data to CSV
```
trainX = pd.DataFrame(X_train)
trainX[label] = y_train
testX = pd.DataFrame(X_test)
testX[label] = y_test
```
```
trainX.to_csv("train-V-1.csv", index=False)
testX.to_csv("test-V-1.csv", index=False)
```

## Upload Data to S3
```
sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer"
trainpath = sess.upload_data(path="train-V-1.csv", bucket=bucket, key_prefix=sk_prefix)
testpath = sess.upload_data(path="test-V-1.csv", bucket=bucket, key_prefix=sk_prefix)

print(trainpath)
print(testpath)
```

## Custom Training Script
The training script (script.py) contains:

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
    
if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()
    
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-1)
    
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()
    
    print("Label column is: ",label)
    print()
    
    print("Data Shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
  
    print("Training RandomForest Model.....")
    print()
    model =  RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose = 3,n_jobs=-1)
    model.fit(X_train, y_train)
    print()
    

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model,model_path)
    print("Model persisted at " + model_path)
    print()

    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep = classification_report(y_test,y_pred_test)

    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)
```

## Model Training
```
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="arn:aws:iam::566373416292:role/service-role/AmazonSageMaker-ExecutionRole-20230120T164209",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="RF-custom-sklearn",
    hyperparameters={
        "n_estimators": 100,
        "random_state": 0,
    },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)
```

## Start Training
```
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True)
```

```

sklearn_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)
```
```
artifact
```

## Model Deployment
### Create Model
```
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = SKLearnModel(
    name=model_name,
    model_data=artifact,
    role="arn:aws:iam::566373416292:role/service-role/AmazonSageMaker-ExecutionRole-20230120T164209",
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION,
)
```

### Deploy Endpoint
```
endpoint_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)
```
```
endpoint_name
```
```

testX[features][0:2].values.tolist()
```
```

print(predictor.predict(testX[features][0:2].values.tolist()))
```


## Cleanup
```
sm_boto3.delete_endpoint(EndpointName=endpoint_name)
```
