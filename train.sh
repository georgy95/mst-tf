#!/usr/bin/env bash

usage () {
  echo "usage: train.sh [local | remote | tune ]

Use 'local' to train locally with a local data file, and 'train' to
run on ML Engine. For ML Engine jobs the train and valid directories must reside on GCS.

Examples:
# train locally
./train.sh local
# train on ML Engine with hparms.py
./train.sh remote
"

}

date
TIME=`date +"%Y%m%d_%H%M%S"`
BUCKET_NAME=YOUR_BUCKET
BUCKET=gs://${BUCKET_NAME}
DATAPATH=gs://${BUCKET_NAME}/data
WEIGHTS=gs://${BUCKET_NAME}/jobs/mst_training_remote_20190524_103506/weights/decoder.h5
LOCAL_WEIGHTS=./trainer/data/weights/weights.h5

if [[ $# < 1 ]]; then
  usage
  exit 1
fi

# set job vars
JOB_TYPE="$1"
EVAL="$2"
JOB_NAME=mst_training_${JOB_TYPE}_${TIME}
export JOB_NAME=${JOB_NAME}
REGION=europe-west1

if [[ ${JOB_TYPE} == "local" ]]; then

  gcloud ml-engine local train \
    --module-name trainer.train \
    --package-path ./trainer \
    -- \
    --datapath trainer/data \
    --job-dir trainer/jobs/${JOB_NAME}/ \
    --weights ${LOCAL_WEIGHTS} \


elif [[ ${JOB_TYPE} == "remote" ]]; then

  gcloud ml-engine jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --job-dir ${BUCKET}/jobs/${JOB_NAME}/ \
    --module-name trainer.train \
    --package-path ./trainer \
    --config trainer/config/config_train.json \
    -- \
    --datapath ${DATAPATH} \
    # --weights ${WEIGHTS} \

else
  usage
fi

