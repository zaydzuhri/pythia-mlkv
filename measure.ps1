python measure_memory.py ../pythia-160m-deduped --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-gqa-48-b12-g2-v2 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-48-b12-g2-v9 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mqa-12-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-12-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-6-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-4-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-2-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;
python measure_memory.py faridlazuarda/pythia-160m-mlkv-1-b12-g2-v1 --revision 'master' --mlkv --log-file measurements.csv;