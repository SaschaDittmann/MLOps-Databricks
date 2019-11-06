# Databricks notebook source
# MAGIC %md ## Serving Models with Microsoft Azure ML
# MAGIC 
# MAGIC ##### NOTE: I do not recommend using *Run All* because it takes several minutes to deploy and update models; models cannot be queried until they are active.

# COMMAND ----------

#dbutils.widgets.removeAll()
#dbutils.widgets.text("model_image_id", "")

# COMMAND ----------

model_image_id = dbutils.widgets.getArgument("model_image_id")
print("Model Image ID:", model_image_id)

# COMMAND ----------

# MAGIC %md ### Create or load an Azure ML Workspace

# COMMAND ----------

# MAGIC %md Before models can be deployed to Azure ML, you must create or obtain an Azure ML Workspace. The `azureml.core.Workspace.create()` function will load a workspace of a specified name or create one if it does not already exist. For more information about creating an Azure ML Workspace, see the [Azure ML Workspace management documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace).

# COMMAND ----------

import azureml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

workspace_name = dbutils.secrets.get(scope = "azureml", key = "workspace_name")
workspace_location = "westeurope"
resource_group = dbutils.secrets.get(scope = "azureml", key = "resource_group")
subscription_id = dbutils.secrets.get(scope = "azureml", key = "subscription_id")

svc_pr = ServicePrincipalAuthentication(
    tenant_id = dbutils.secrets.get(scope = "azureml", key = "tenant_id"),
    service_principal_id = dbutils.secrets.get(scope = "azureml", key = "client_id"),
    service_principal_password = dbutils.secrets.get(scope = "azureml", key = "client_secret"))

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth=svc_pr,
                             exist_ok=True)

# COMMAND ----------

# MAGIC %md ## Deploy the model to "dev" using [Azure Container Instances (ACI)](https://docs.microsoft.com/en-us/azure/container-instances/)
# MAGIC 
# MAGIC The [ACI platform](https://docs.microsoft.com/en-us/azure/container-instances/) is the recommended environment for staging and developmental model deployments.

# COMMAND ----------

# MAGIC %md ### Create an ACI webservice deployment using the model's Container Image
# MAGIC 
# MAGIC Using the Azure ML SDK, deploy the Container Image for the trained MLflow model to ACI.

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.image import Image

model_image = Image(workspace, id=model_image_id)

dev_webservice_name = "wine-quality-aci"
dev_webservice_deployment_config = AciWebservice.deploy_configuration()
dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=workspace, deployment_target=None, overwrite=True)

dev_webservice.wait_for_deployment()

# COMMAND ----------

# MAGIC %md ## Query the deployed model in "dev"

# COMMAND ----------

# MAGIC %md ### Load dataset

# COMMAND ----------

import numpy as np
import pandas as pd

csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
  data = pd.read_csv(csv_url, sep=';')
except Exception as e:
  logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

data = data.drop(["quality"], axis=1)[:10]

# COMMAND ----------

# MAGIC %md ## Create sample input vector

# COMMAND ----------

query_input = data.to_json(orient='split')
query_input = eval(query_input)
query_input.pop('index', None)

# COMMAND ----------

# MAGIC %md #### Evaluate the sample input vector by sending an HTTP request
# MAGIC Query the ACI webservice's scoring endpoint by sending an HTTP POST request that contains the input vector.

# COMMAND ----------

import requests
import json

def query_endpoint_example(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=json.dumps(inputs), headers=headers)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

print("Webservice URL:", dev_webservice.scoring_uri)

# COMMAND ----------

dev_prediction = query_endpoint_example(scoring_uri=dev_webservice.scoring_uri, inputs=query_input)