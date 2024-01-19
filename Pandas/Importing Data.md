# Import data
- Flat files, e.g. .txts, .csvs
- Files from other software
- Relational databases

# Reading a text file
```python
filename = 'name.txt'
file = open(filename, mode='r') # 'r' is to read / 'w' is to write
text = file.read()
print(text)

file.close()
```
Context manager with
```python
with open('name.txt', 'r') as file:
  print(file.read())
```
Flat files
- Text files containing records
- Table data
- Record: row of fields or attributes
- Column: feature or attribute

File extensions
- .csv: comma separated values
- .txt: text file
- commas, tabs: delimiters

# Importing flat files using NumPy
- NumPy arrays: standard for storing numerical data
- Essential for other packages: e.g., scikit-learn
- loadtxt()
- genfromtxt()
  
```python
import numpy as np

filename = 'name.txt'
data = np.loadtxt(filename, delimiter= ',')
```
Customizing your NumPy import
```python
import numpy as np

filename = 'name.txt'
data = np.loadtxt(filename, delimiter= ',', skiprows=1, usecols=[0,2], dtype=str)

print(data)
```

# Importing flat files using Pandas
```python
import pandas as pd

filename = 'name.csv'
data = pd.read_csv(filename)
data.head()

data_array = data.values
```

# Other file types
- Excel spreadsheets
- MATLAB files
- SAS files
- Stata files
- HDF5 files
- Pickled files
  - File type native to Python
  - Motivation: many datatypes for which it isn't obvious how to store them
  - Pickled files are serialized
  - Serialize = convert object to bytestream

## Pickled files
```python
import pickle

with open('name.pkl', 'rb') as file:
  data = pickle.load(file)

print(data)
```
## Importing Excel spreadsheets
```python
import pandas as pd
file = 'name.xlsx'
data = pd.ExcelFile(file) # or pd.read_excel

print(data.sheet_names)

df1 = data.parse('sheet name') # sheet name as a string
df2 = data.parse(0) # sheet index as a float
```
## Importing SAS/Stata files
- SAS: Statistical Analysis System
- Stata: "Statistics" + "data"
- SAS: business analytics and biostatistics
- Stata: academic social sciences research

SAS files
- Used for:
  - Advanced analytics
  - Multivariate analysis
  - Business intelligence
  - Data management
  - Predictive analytics
  - Standard for computational analysis

Importing SAS files
```python
import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT('name.sas7bdat') as file:
  df_sas = file.to_data_frame()
```
Importing Stata files
```python
import pandas as pd
data = pd.read_stata('name.dta')
```
## Importing HDF5 files
- Hierarchical Data Format version 5
- Standard for storing large quantities of numerical data
- Datasets can be hundeds of gigabytes or terabytes
- HDF5 can scale to exabytes

```python
import h5py

filename = 'name.hdf5'
data = h5py.File(filename, 'r') 

print(type(data))
```

Structure of HDF5 files
```python
for key in data.keys():
  print(key) # meta, quality and strain

print(type(data['meta']))
```

```python
for key in data['meta'].keys():
  print(key)

print( np.array(data['meta']['Description']), np.array(data['meta']['Detector']) )
```

## Importing MATLAB files
- "Matrix Laboratory"
- Industry standard in engineering and science
- Data saved as .mat files
- scipy.io.loadmat(): read .mat files
- scipy.io.savemat(): write .mat files

```python
import scipy.io

filename = 'name.mat'
mat = scipy.io.loadmat(filename)

print(type(mat)) # class dict
"""
keys = Matlab variable names
values = objects assigned to variables
"""

print(type(mat['x'])) # class numpy.ndarray
```

# Creating a database engine in Python
- SQLite database
  - Fast and simple
- SQLAlchemy
  - Works with many Relational Database Management Systems (RDMS)
```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///name.sqlite')
```

Getting table names
```python
table_names = engine.table_names()

print(table_names)
```

## Querying relational databases
- Basic SQL query
```sql
SELECT * FROM Table_Name
```
- Returns all columns of all rows of the table
- Example:
```sql
SELECT * FROM orders
```
Workflow of SQL querying
- Import packages and functions
- Create the database engine
- Connect to the engine
- Query the database
- Save query results to a DataFrame
- Close the connection
  
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///name.sqlite')
con = engine.connect()
rs. con.execute('query')
df = pd.DataFrame(rs.fetchall())
con.close()

print(df.head())
```
Set the DataFrame column names
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///name.sqlite')
con = engine.connect()
rs. con.execute('query')
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

print(df.head())
```
Using the context manager
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///name.sqlite')

with engine.connect() as con:
  rs = con.execute('query')
  df = pd.DataFrame(rs.fetchmany(size=5))
  df.columns = rs.keys()
```
## Querying relational databases directly with Pandas
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///name.sqlite')

df = pd.read_sql_query('query', engine)
```
## Advanced querying: exploiting table relationships
INNER JOIN in Python (pandas)
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///name.sqlite')
df = pd.read_sql_query('SELECT table1, table2 FROM table1 INNER JOIN table2 ON table1.column = table2.column', engine)

print(df.head())
```
