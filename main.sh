# Entry point to running supervised experiments on SDGi Corpus

export experiment=replication
export models="bow_svm ada_svm bow_mlp ada_mlp gnn"

for model in $models; do

  # in-domain testing
  python main.py --size s --language en --model $model --experiment $experiment
  python main.py --size m --language en --model $model --experiment $experiment
  python main.py --size l --language en --model $model --experiment $experiment
  python main.py --size x --language en --model $model --experiment $experiment
  python main.py --size x --language fr --model $model --experiment $experiment
  python main.py --size x --language es --model $model --experiment $experiment
  python main.py --size x --language xx --model $model --experiment $experiment

  # out-of-domain testing
  python main.py --size x --language en --model $model --experiment $experiment --ood
  python main.py --size x --language xx --model $model --experiment $experiment --ood
done
