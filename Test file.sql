USE catalog_40_copper_post16_assurance_analysis.sql_translation
GO
                        
SELECT employee_id, first_name, last_name, department_id, hire_date
INTO #employees
FROM employees
WHERE hire_date >= '1993-01-01'
                        
SELECT * 
FROM departments
INNER JOIN #employees ON employees.department_id = department.department_id
ORDER BY hire_date