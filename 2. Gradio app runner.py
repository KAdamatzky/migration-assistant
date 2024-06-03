# Databricks notebook source
# MAGIC %pip install openai --upgrade

# COMMAND ----------

# MAGIC %pip install sqlglot openpyxl dbtunnel[asgiproxy,gradio] databricks-vectorsearch

# COMMAND ----------

# MAGIC %pip install -U gradio==4.27.0 databricks-sdk databricks-sql-connector

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from utils.configloader import configLoader
cl = configLoader() 
cl.read_yaml_to_env("config.yaml")

# COMMAND ----------

import os
print(os.environ["MAX_TOKENS"])

# COMMAND ----------

from dbtunnel import dbtunnel
dbtunnel.kill_port(8080)
app='././1. gradio_app.py'
dbtunnel.gradio(path=app).run()

# COMMAND ----------

# Temp tables, variable declarations, date formatting, and square brackets
"""
USE [catalog_40_copper_post16_assurance_analysis].[sql_translation]
GO

DECLARE @Location INT
SET @Location = 1700

SELECT [employee_id], [first_name], [last_name], [department_id], [hire_date]
INTO #employee_temp
FROM [employees]
WHERE [hire_date] >= '1993-01-02' 


SELECT * 
FROM [departments]
WHERE location_id = @Location
INNER JOIN #employee_temp ON [employees].[department_id] = [department].[department_id]
ORDER BY [hire_date]
"""

# COMMAND ----------

# Similar code
"""
USE catalog_40_copper_post16_assurance_analysis.sql_translation
GO

SELECT employee_id, first_name, last_name, department_id, hire_date
INTO #employee_temp
FROM employees
WHERE hire_date >= '1999-03-01' 


SELECT * 
FROM departments
INNER JOIN #employee_temp ON employees.department_id = department.department_id
ORDER BY hire_date
"""

# COMMAND ----------

# Temporary table example
"""
USE catalog_40_copper_post16_assurance_analysis.sql_translation

DROP TABLE IF EXISTS #temp1

SELECT * 
INTO #temp1
FROM employees

SELECT TOP(100) * 
FROM #temp1
"""

# COMMAND ----------

"""

SELECT
  c.[country_name],
  AVG([dep_count]) AS average_dependents
FROM
  (
    SELECT
      e.[employee_id]
      ,e.[department_id]
      ,COUNT(d.[dependent_id]) AS dep_count
    FROM
      [catalog_40_copper_post16_assurance_analysis].[sql_translation].[employees] e
      LEFT JOIN [catalog_40_copper_post16_assurance_analysis].[sql_translation].[dependents] d ON e.[employee_id] = d.[employee_id]
    GROUP BY
      e.[employee_id]
      ,e.[department_id]
  ) AS subquery
  JOIN [catalog_40_copper_post16_assurance_analysis].[sql_translation].[departments] dep ON subquery.[department_id] = dep.[department_id]
  JOIN [catalog_40_copper_post16_assurance_analysis].[sql_translation].[locations] l ON dep.[location_id] = l.[location_id]
  JOIN [catalog_40_copper_post16_assurance_analysis].[sql_translation].[countries] c ON l.[country_id] = c.[country_id]
GROUP BY
  c.[country_name]
ORDER BY
  c.[country_name]


"""

# COMMAND ----------

"""

SELECT
  c.country_name,
  AVG(d.num_dependents) AS avg_dependents,
  AVG(e.salary) AS avg_salary
FROM
  (
    SELECT
      employee_id,
      COUNT(dependent_id) AS num_dependents
    FROM
      catalog_40_copper_post16_assurance_analysis.sql_translation.dependents
    GROUP BY
      employee_id
  ) d
  RIGHT JOIN catalog_40_copper_post16_assurance_analysis.sql_translation.employees e ON d.employee_id = e.employee_id
  JOIN catalog_40_copper_post16_assurance_analysis.sql_translation.departments dep ON e.department_id = dep.department_id
  JOIN catalog_40_copper_post16_assurance_analysis.sql_translation.locations l ON dep.location_id = l.location_id
  JOIN catalog_40_copper_post16_assurance_analysis.sql_translation.countries c ON l.country_id = c.country_id
GROUP BY
  c.country_name
ORDER BY
  c.country_name

"""

# COMMAND ----------

"""
with average_salarys as (
  SELECT
    d.department_id,
    avg(salary) as average_salary
  FROM catalog_40_copper_post16_assurance_analysis.sql_translation.employees e
  inner join catalog_40_copper_post16_assurance_analysis.sql_translation.departments d on e.department_id = d.department_id
  group by d.department_id
)

select 
  first_name
  ,last_name
  ,salary
  ,average_salary 
  ,department_name
from catalog_40_copper_post16_assurance_analysis.sql_translation.employees e
inner join average_salarys a on e.department_id = a.department_id
inner join catalog_40_copper_post16_assurance_analysis.sql_translation.departments d on e.department_id = d.department_id
where salary > a.average_salary
"""
