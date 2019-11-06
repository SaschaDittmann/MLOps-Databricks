# Databricks notebook source
# MAGIC %md ## Serving Models with Microsoft Azure ML
# MAGIC 
# MAGIC ##### NOTE: I do not recommend using *Run All* because it takes several minutes to deploy and update models; models cannot be queried until they are active.

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

# MAGIC %md ## Build an Azure Container Image for model deployment

# COMMAND ----------

# MAGIC %md ### Use MLflow to build a Container Image for the trained model
# MAGIC 
# MAGIC Use the `mlflow.azuereml.build_image` function to build an Azure Container Image for the trained MLflow model. This function also registers the MLflow model with a specified Azure ML workspace. The resulting image can be deployed to Azure Container Instances (ACI) or Azure Kubernetes Service (AKS) for real-time serving.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

experiment_name = "/Shared/WineQuality"
experiment = MlflowClient().get_experiment_by_name(experiment_name)
experiment_ids = eval('[' + experiment.experiment_id + ']')
# all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
print("Experiment IDs:", experiment_ids)

query = "metrics.rmse < 0.8"
runs = MlflowClient().search_runs(experiment_ids, query, ViewType.ALL)

rmse_low = None
run_id = None
for run in runs:
  if (rmse_low == None or run.data.metrics['rmse'] < rmse_low):
    rmse_low = run.data.metrics['rmse']
    run_id = run.info.run_id
print("Lowest RMSE:", rmse_low)
print("Run ID:", run_id)

model_uri = "runs:/" + run_id + "/model"

# COMMAND ----------

import mlflow.azureml

model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri, 
                                                      workspace=workspace,
                                                      model_name="winequality",
                                                      image_name="winequality",
                                                      description="Sklearn ElasticNet image for predicting wine quality",
                                                      synchronous=False)

model_image.wait_for_creation(show_output=True)

# COMMAND ----------

dbutils.notebook.exit('{"model_image_id": "%s"}' % model_image.id)