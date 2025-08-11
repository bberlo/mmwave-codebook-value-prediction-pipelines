#!/bin/bash

rm -rf results/*
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n infonce -t leave-out -do_t random -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t random -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t orientation -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t orientation -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t position -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t position -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t radar -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t leave-out -do_t radar -h_a 1 -g 0 -s 5481

python experiment_automation.py -m_n minirocket -t leave-out -do_t random -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t random -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t orientation -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t orientation -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t position -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t position -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t radar -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t leave-out -do_t radar -h_a 1 -g 0 -s 5481

python experiment_automation.py -m_n ompursuit -t leave-out -do_t random -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t random -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t orientation -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t orientation -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t position -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t position -h_a 1 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t radar -h_a 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t leave-out -do_t radar -h_a 1 -g 0 -s 5481

mkdir results/leave-out
mv results/*.csv results/leave-out
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n infonce -t frame-ablation -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t frame-ablation -do_t orientation -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t frame-ablation -do_t position -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n infonce -t frame-ablation -do_t radar -h_a 0 -cv_s 0 -g 0 -s 5481

python experiment_automation.py -m_n minirocket -t frame-ablation -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t frame-ablation -do_t orientation -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t frame-ablation -do_t position -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t frame-ablation -do_t radar -h_a 0 -cv_s 0 -g 0 -s 5481

python experiment_automation.py -m_n ompursuit -t frame-ablation -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t frame-ablation -do_t orientation -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t frame-ablation -do_t position -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n ompursuit -t frame-ablation -do_t radar -h_a 0 -cv_s 0 -g 0 -s 5481

mkdir results/frame-ablation
mv results/*.csv results/frame-ablation
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n infonce -t shapley -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481
python experiment_automation.py -m_n minirocket -t shapley -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481

mkdir results/shapley-importance
mv results/*.csv results/shapley-importance
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n infonce -t remove-retrain -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481 -r_p results/shapley-importance/infonce_*.csv
python experiment_automation.py -m_n minirocket -t remove-retrain -do_t random -h_a 0 -cv_s 0 -g 0 -s 5481 -r_p results/shapley-importance/minirocket_*.csv

mkdir results/remove-retrain
mv results/*.csv results/remove-retrain
rm -rf Streams/*
rm -rf tmp/*
