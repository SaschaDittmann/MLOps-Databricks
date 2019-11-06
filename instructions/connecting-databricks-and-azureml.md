# Connecting Azure Databricks with the Azure ML Service Workspace

## Step 1: Create Azure AD Service Principal

``` bash
az ad sp create-for-rbac -n "http://MLOps-Databricks"
```

> NOTE: If you want, you can narrow that down to a specific resource group use the following command

``` bash
az ad sp create-for-rbac -n "http://MLOps-Databricks" --role contributor --scopes /subscriptions/{SubID}/resourceGroups/{ResourceGroup1}
```

## Step 2: Install / Update Databricks CLI

``` bash
pip install -U databricks-cli
```

> NOTE: You need python 2.7.9 or later / 3.6 or later to install and use the Databricks command-line interface (CLI) 

## Step 3: Create Databricks Secrets Scope

``` bash
databricks secrets create-scope --scope azureml
```

## Step 4: Add Databricks Secrets

``` bash
# Use the "tenant" property from the Azure AD Service Principal command output
databricks secrets put --scope azureml --key tenant_id
# Use the "appId" property from the Azure AD Service Principal command output
databricks secrets put --scope azureml --key client_id
# Use the "password" property from the Azure AD Service Principal command output
databricks secrets put --scope azureml --key client_secret
databricks secrets put --scope azureml --key subscription_id
databricks secrets put --scope azureml --key resource_group
databricks secrets put --scope azureml --key workspace_name
```
