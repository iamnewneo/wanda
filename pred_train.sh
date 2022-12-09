export SKLEARNEX_VERBOSE=INFO
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export ENV="prod"
echo "$(date)" | tee -a out_pred_train_log.log
python -u -m wanda.prednet_driver 2>&1 | tee -a out_pred_train_log.log