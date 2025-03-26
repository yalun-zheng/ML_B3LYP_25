cd PATH_TO_CODE
mkdir -p data model data/xyz data/tmp data/pkl data/pkl/G2  data/pkl/test

export PYSCF_TMPDIR="`pwd`/data/tmp"
export PYSCF_MAX_MEMORY=400000

python src/train_fcn.py cfg.yaml
python src/valid_fcn.py cfg.yaml
python src/test_fcn.py cfg.yaml

# rm -r data/tmp
