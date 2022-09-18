echo "$(date)" | tee -a out_train_log.log
python -u -m wanda.driver 2>&1 | tee -a out_train_log.log