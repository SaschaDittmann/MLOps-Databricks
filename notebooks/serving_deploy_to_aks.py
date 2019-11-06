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

# MAGIC %md ## Deploy the model to production using [Azure Kubernetes Service (AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/).

# COMMAND ----------

# MAGIC %md ### Create a new AKS cluster
# MAGIC 
# MAGIC If you do not have an active AKS cluster for model deployment, create one using the Azure ML SDK.

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

aks_name = 'myaks'

try:
  aks_target = AksCompute(workspace, name=aks_name)
  print('found existing:', aks_target.name)
except ComputeTargetException:
  print('creating new.')

  # AKS configuration
  prov_config = AksCompute.provisioning_configuration(
    agent_count=3,
    vm_size="Standard_B4ms"
  )

  # Create the cluster
  aks_target = ComputeTarget.create(
    workspace, 
    name = aks_name, 
    provisioning_configuration = prov_config
  )
  
  # Wait for the create process to complete
  aks_target.wait_for_completion(show_output = True)
  print(aks_target.provisioning_state)
  print(aks_target.provisioning_errors)

# COMMAND ----------

# MAGIC %md ### Deploy to the model's image to the specified AKS cluster

# COMMAND ----------

from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import Image

# Get Model
model_image = Image(workspace, id=model_image_id)

# Get Webservice
prod_webservice_name = "wine-quality-aks"
try:
  prod_webservice = Webservice(workspace, prod_webservice_name)
  print('updating existing webservice.')
  prod_webservice.update(image=model_image)
  prod_webservice.wait_for_deployment(show_output = True)
except:
  print('creating new webservice.')
  # Set configuration and service name
  prod_webservice_deployment_config = AksWebservice.deploy_configuration()
  # Deploy from image
  prod_webservice = Webservice.deploy_from_image(workspace = workspace, 
                                                 name = prod_webservice_name,
                                                 image = model_image,
                                                 deployment_config = prod_webservice_deployment_config,
                                                 deployment_target = aks_target)
  # Wait for the deployment to complete
  prod_webservice.wait_for_deployment(show_output = True)

# COMMAND ----------

# MAGIC %md ## Query the deployed model in production

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

# MAGIC %md ### Create sample input vector

# COMMAND ----------

query_input = data.to_json(orient='split')
query_input = eval(query_input)
query_input.pop('index', None)

# COMMAND ----------

# MAGIC %md #### Evaluate the sample input vector by sending an HTTP request
# MAGIC Query the AKS webservice's scoring endpoint by sending an HTTP POST request that includes the input vector. The production AKS deployment may require an authorization token (service key) for queries. Include this key in the HTTP request header.

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

prod_scoring_uri = prod_webservice.scoring_uri
prod_service_key = prod_webservice.get_keys()[0] if len(prod_webservice.get_keys()) > 0 else None
print("Webservice URL:", prod_scoring_uri)

# COMMAND ----------

prod_prediction1 = query_endpoint_example(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=query_input)