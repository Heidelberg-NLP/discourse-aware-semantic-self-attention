# Eval trained models

```

declare -a model_names_list=(
baseline_bidaf.tar.gz
baseline_qanet.tar.gz
dassa_coref_feats.tar.gz
dassa_multi_srl_sdp_exp.tar.gz
dassa_multi_srl_sdp_exp_coref_feats.tar.gz
dassa_multi_srl_sdp_exp_nonexp.tar.gz
dassa_multi_srl_sdp_nonexp.tar.gz
dassa_sdp_exp.tar.gz
dassa_sdp_exp_nonexp.tar.gz
dassa_sdp_exp_nosense.tar.gz
dassa_sdp_ne.tar.gz
dassa_sdp_ne_nosense.tar.gz
dassa_sentspan3.tar.gz
dassa_srl_3verbs.tar.gz
dassa_srl_4verbs.tar.gz
)

# Indicates if we want to evaluate the eval input with the trained model.
EVAL_PREDICTIONS=TRUE

# Indicates if we want to calculate different metrics such as question types/lengths etc.
GENERATE_METRICS=TRUE

# If we have started evaluation but it failed and we want to append the failed example predictions
SKIP_OR_APPEND_FILE=TRUE

for model_name_curr in "${model_names_list[@]}"
do
    JOB_NAME=${model_name_curr}

    BATCH_SIZE=1
    python_exec=/home/mitarb/mihaylov/anaconda3/envs/dassa/bin/python
    ARCHIVE_FILE=trained_models/emnlp_paper/${model_name_curr}


    OUTPUT_DIR=_output/emnlp_paper/${JOB_NAME}
    mkdir -p ${OUTPUT_DIR}
    mkdir -p _jobs

    cp logs/job_${JOB_NAME}.log logs/job_${JOB_NAME}_train.log

    RESEARCH_DIR=/home/mitarb/mihaylov/research

    CUDA_DEVICE=0

    JOB_NAME=eval_${JOB_NAME}
    JOB_BASH_FILE=_jobs/${JOB_NAME}.sh

    TRAIN_FILE=data/narrativeqa_annotated/summaries_annotated.jsonl.train
    DEV_FILE=data/narrativeqa_annotated/summaries_annotated.jsonl.valid
    TEST_FILE=data/narrativeqa_annotated/summaries_annotated.jsonl.test
    TEST2_FILE=

    TRAIN_EVAL_OUT=predictions_train.json
    DEV_EVAL_OUT=predictions_dev.json
    TEST_EVAL_OUT=predictions_test.json
    TEST2_EVAL_OUT=

    TRAIN_EVAL_ANNO=
    DEV_EVAL_ANNO=
    TEST_EVAL_ANNO=tools/annotations/narrativeqa-annotation.json
    TEST2_EVAL_ANNO=

    DEV_EVAL_NUM=3461
    TEST_EVAL_NUM=10557
    TEST2_EVAL_NUM=

    ### Create BASH file
    ### We create bash file with the required evaluation and execute it!
    echo -e "#Script for ${JOB_NAME}\n" > ${JOB_BASH_FILE}

     # Predict DEV
     EVAL_FILE=${DEV_FILE}
     ANNOTATION_FILE=${DEV_EVAL_ANNO}
     if [ -n "${ANNOTATION_FILE}" ]; then
        ANNOTATION_FILE=" -a ${ANNOTATION_FILE}"
     fi
     PREDICTIONS_OUT_FILE=${OUTPUT_DIR}/${DEV_EVAL_OUT}
     EVAL_NUM=${DEV_EVAL_NUM}
     if [ "${EVAL_PREDICTIONS}" == "TRUE" ]; then
         START_ID=-1
         END_ID=-1
         FILE_OPEN_MODE=w
         SKIP_CURR=FALSE
         EVAL_FILE_LEN=$(wc -l < "${EVAL_FILE}")
         if [ -f ${PREDICTIONS_OUT_FILE} ]; then
            if [ "${SKIP_OR_APPEND_FILE}" == "TRUE" ]; then
                START_ID=$(wc -l < "${PREDICTIONS_OUT_FILE}")
                if [ "${EVAL_NUM}" == "${START_ID}" ]; then
                    SKIP_CURR=TRUE
                fi
                FILE_OPEN_MODE=a
            fi
         fi

         if [ "${SKIP_CURR}" == "FALSE" ]; then
             JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} docqa/run.py evaluate_custom --archive_file ${ARCHIVE_FILE} --evaluation_data_file ${EVAL_FILE} --output_file ${PREDICTIONS_OUT_FILE} --cuda_device ${CUDA_DEVICE} --batch_size=${BATCH_SIZE} --start_id ${START_ID} --end_id ${END_ID} --file_open_mode=${FILE_OPEN_MODE}"
             echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
             echo -e "\n" >> ${JOB_BASH_FILE}
         fi
     fi

     # DEV generate metrics
     if [ "${GENERATE_METRICS}" == "TRUE" ]; then
         JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} tools/narrativeqa_eval_generation.py -i ${PREDICTIONS_OUT_FILE} ${ANNOTATION_FILE}"
         echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
         echo -e "\n" >> ${JOB_BASH_FILE}
     fi

     # Predict TEST
     EVAL_FILE=${TEST_FILE}
     ANNOTATION_FILE=${TEST_EVAL_ANNO}
     if [ -n "${ANNOTATION_FILE}" ]; then
        ANNOTATION_FILE=" -a ${ANNOTATION_FILE}"
     fi
     PREDICTIONS_OUT_FILE=${OUTPUT_DIR}/${TEST_EVAL_OUT}
     EVAL_NUM=${TEST_EVAL_NUM}
     if [ "${EVAL_PREDICTIONS}" == "TRUE" ]; then
         START_ID=-1
         END_ID=-1
         FILE_OPEN_MODE=w
         SKIP_CURR=FALSE
         EVAL_FILE_LEN=$(wc -l < "${EVAL_FILE}")
         if [ -f ${PREDICTIONS_OUT_FILE} ]; then
            if [ "${SKIP_OR_APPEND_FILE}" == "TRUE" ]; then
                START_ID=$(wc -l < "${PREDICTIONS_OUT_FILE}")
                if [ "${EVAL_NUM}" == "${START_ID}" ]; then
                    SKIP_CURR=TRUE
                fi
                FILE_OPEN_MODE=a
            fi
         fi

         if [ "${SKIP_CURR}" == "FALSE" ]; then
             JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} docqa/run.py evaluate_custom --archive_file ${ARCHIVE_FILE} --evaluation_data_file ${EVAL_FILE} --output_file ${PREDICTIONS_OUT_FILE} --cuda_device ${CUDA_DEVICE} --batch_size=${BATCH_SIZE} --start_id ${START_ID} --end_id ${END_ID} --file_open_mode=${FILE_OPEN_MODE}"
             echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
             echo -e "\n" >> ${JOB_BASH_FILE}
         fi
     fi

     # Test generate metrics
     if [ "${GENERATE_METRICS}" == "TRUE" ]; then
         JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} tools/narrativeqa_eval_generation.py -i ${PREDICTIONS_OUT_FILE} ${ANNOTATION_FILE}"
         echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
         echo -e "\n" >> ${JOB_BASH_FILE}
     fi

    # Predict TEST2
    if [ "$TEST2_FILE" -ne "" ]; then
        EVAL_FILE=${$TEST2_FILE}
        ANNOTATION_FILE=${TEST2_EVAL_ANNO}
        if [ -n "${ANNOTATION_FILE}" ]; then
            ANNOTATION_FILE=" -a ${ANNOTATION_FILE}"
        fi
        PREDICTIONS_OUT_FILE=${OUTPUT_DIR}/${TEST2_EVAL_OUT}
        EVAL_NUM=${TEST2_EVAL_NUM}
        if [ "${EVAL_PREDICTIONS}" == "TRUE" ]; then
            START_ID=-1
            END_ID=-1
            FILE_OPEN_MODE=w
            SKIP_CURR=FALSE
            EVAL_FILE_LEN=$(wc -l < "${EVAL_FILE}")
            if [ -f ${PREDICTIONS_OUT_FILE} ]; then
               if [ "${SKIP_OR_APPEND_FILE}" == "TRUE" ]; then
                   START_ID=$(wc -l < "${PREDICTIONS_OUT_FILE}")
                   if [ "${EVAL_NUM}" == "${START_ID}" ]; then
                       SKIP_CURR=TRUE
                   fi
                   FILE_OPEN_MODE=a
               fi
            fi

            if [ "${SKIP_CURR}" == "FALSE" ]; then
                JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} docqa/run.py evaluate_custom --archive_file ${ARCHIVE_FILE} --evaluation_data_file ${EVAL_FILE} --output_file ${PREDICTIONS_OUT_FILE} --cuda_device ${CUDA_DEVICE} --batch_size=${BATCH_SIZE} --start_id ${START_ID} --end_id ${END_ID} --file_open_mode=${FILE_OPEN_MODE}"
                echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
                echo -e "\n" >> ${JOB_BASH_FILE}
            fi
        fi

        # Test generate metrics
        if [ "${GENERATE_METRICS}" == "TRUE" ]; then
            JOB_SCRIPT="RESEARCH_DIR=${RESEARCH_DIR};PYTHONPATH=. ${python_exec} tools/narrativeqa_eval_generation.py -i ${PREDICTIONS_OUT_FILE} ${ANNOTATION_FILE}"
            echo ${JOB_SCRIPT} >> ${JOB_BASH_FILE}
            echo -e "\n" >> ${JOB_BASH_FILE}
        fi
    fi


    echo "bash ${JOB_BASH_FILE}"
    bash ${JOB_BASH_FILE}

done

```
# Export results
```
bash tools/display_out_metrics_narrativeqa.sh
```
