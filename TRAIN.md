# Train models

```

mkdir -p _trained_models/
mkdir -p _output/

# set params
declare -a config_files_list=(
training_config/qa/span_prediction/narrativeqa/baseline_bidaf.json
training_config/qa/span_prediction/narrativeqa/baseline_qanet.json
training_config/qa/span_prediction/narrativeqa/dassa_coref_feats.json
training_config/qa/span_prediction/narrativeqa/dassa_multi_srl_sdp_exp.json
training_config/qa/span_prediction/narrativeqa/dassa_multi_srl_sdp_exp_coref_feats.json
training_config/qa/span_prediction/narrativeqa/dassa_multi_srl_sdp_exp_nonexp.json
training_config/qa/span_prediction/narrativeqa/dassa_multi_srl_sdp_nonexp.json
training_config/qa/span_prediction/narrativeqa/dassa_sdp_exp.json
training_config/qa/span_prediction/narrativeqa/dassa_sdp_exp_nonexp.json
training_config/qa/span_prediction/narrativeqa/dassa_sdp_exp_nosense.json
training_config/qa/span_prediction/narrativeqa/dassa_sdp_ne.json
training_config/qa/span_prediction/narrativeqa/dassa_sdp_ne_nosense.json
training_config/qa/span_prediction/narrativeqa/dassa_sentspan3.json
training_config/qa/span_prediction/narrativeqa/dassa_srl_3verbs.json
training_config/qa/span_prediction/narrativeqa/dassa_srl_4verbs.json
)

# set params
declare -a config_files_list=(
training_config/qa/span_prediction/narrativeqa/dassa_sdp_exp.json
)

BATCH_SIZE=2
ACCUMULATION_STEPS=16

# set this param with the python file from your `dassa` conda environment
python_exec=/home/mitarb/mihaylov/anaconda3/envs/dassa/bin/python

for config_curr in "${config_files_list[@]}"
do
    CONFIG_FILE=${config_curr}

    JOB_NAME=${CONFIG_FILE##*/}

    JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)
    serialization_dir=_trained_models/${JOB_NAME}

    # run the job
    JOB_SCRIPT="BATCH_SIZE=${BATCH_SIZE} ACCUMULATION_STEPS=${ACCUMULATION_STEPS} PYTHONPATH=. ${python_exec} docqa/run.py train ${CONFIG_FILE} -s ${serialization_dir}"

    #echo "rm -r ${serialization_dir}"
    echo ${JOB_SCRIPT}
    BATCH_SIZE=${BATCH_SIZE} ACCUMULATION_STEPS=${ACCUMULATION_STEPS} PYTHONPATH=. ${python_exec} docqa/run.py train ${CONFIG_FILE} -s ${serialization_dir}
done

```

# Display results

```
# Display results from all runs using the metrics.json from the trained models. See the Note above!
# Note that the rouge implementation version used during training gives higher than actual results.
# The actual results from the paper (using pycocoeval) are obtained in EVALUATE.md

bash tools/display_metrics_narrativeqa.sh
```