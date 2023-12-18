## Import

```python
from google.cloud import bigquery
```

### Client Object
The first step in the workflow is to create a Client object. As you'll soon see, this Client object will play a central role in retrieving information from BigQuery datasets.
```python
# Create a "Client" object
client = bigquery.Client()
```
