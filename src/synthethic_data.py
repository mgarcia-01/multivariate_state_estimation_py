import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

model_data = "test"
file = 'data/'+model_data+'.csv'
data = pd.read_csv(file)
#data = pd.read_csv('data/training.csv')

metadata = Metadata.detect_from_dataframe(data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=100)
#metadata.save_to_json('metadata.json')

# in the future, you can reload the metadata object from the file
#metadata = Metadata.load_from_json('metadata.json')
#synthetic_data.to_csv("data/syn_test.csv")
out_data = 'data/syn_'+model_data+'.csv'
synthetic_data.to_csv(out_data, index = False)
#synthetic_data.to_csv("data/syn_train.csv", index = False)