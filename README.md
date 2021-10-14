# Getting Started

This project structure template is mean to be used in Data Science and Analytics projects.
Inspired in [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)

## Secrets
Create (and gitignore) a .env file and set them as environment variables with: [dot-env](https://pypi.org/project/python-dotenv/)
* LOCAL_PATH = local path to repo root folder

## Read/Write data
`src/data_api` contains utilities to read/write data to several sources/sinks. These `sources/sinks` are an abstraction layer that take care of the i/o task but do not define the actual source/sink
This **actual soure/sink** is defined by `connections` where a python module is created for each kind (e.g. local.py). In order set a connection, modifiy in `conf.yaml` the kind key in input/output  (setting the same name as the refered module)

Example, setting local input and az_blob output:  
<pre>
├──src  
|  ├──data_api    
|    ├──connections  
|      ├──local.py
|      ├──az_blob.py  
</pre>

In `conf.yaml`:
```yaml
input:
	kind: local
	
output:
	kind: az_blob
```
In order to add more connections, follow these instructions in `get_connection()`:
1. Create a new kind in `connections/__init__.py`
2. Also, create appropieate connection metadata in `connections/__init__.py`
3. Add connection metadata in connection setting

## Running notebooks
In order to be able to import project packages from `./notebooks` folder, it is necessary to add project's path to system paths.
This is set in a module named: `nb_config.py` that also carries general notebook configuration

<pre>
├──src  
|  ├──my_module.py  
|──notebooks  
|  ├──my_notebook.ipynb  
├──nb_config.py  
</pre>

In `nb_config.py`:
```python
root_path = os.environ.get("LOCAL_PATH")
```

In `my_notebook.ipynb`:
```python
%run ../nb_config.py
```
After this script has added project's path, imports can be made from src
```python
from src import my_module
```

## Naming Convetions
### Notebooks
Use XYZZ_{prefix}_{title}_{suffix}
* XYZZ are numbers to identify and sort notebooks
* prefix: any particle that identifies the notebook objective: loadData, dataPrep, modBuild, modEval
* title: notebook title, like salesModel
* suffix: (optional)



# Usefull Resources:
* [Azure devops markdown guide](https://docs.microsoft.com/en-us/azure/devops/project/wiki/markdown-guidance?view=azure-devops#links)
* [README guide](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)