#!/bin/bash
# Runs every evaluation in the report. GIFs go to images/, accuracies to output/results.txt.
# usage: bash run_experiments.sh "<cls indices>" "<seg indices>"
set -e
CLS_I=$1
SEG_I=$2
mkdir -p output images
acc() { grep "test accuracy" | awk '{print $3}'; }

{
echo "== Q1 / Q2 =="
python eval_cls.py --i $CLS_I --output_dir images --exp_name q1
python eval_seg.py --i $SEG_I --output_dir images --exp_name q2

echo "== Q3 rotation (degrees, cls acc, seg acc) =="
for r in 0 15 30 45 60 90 180; do
    echo "$r $(python eval_cls.py --rot $r --output_dir output | acc) $(python eval_seg.py --rot $r --output_dir output | acc)"
done
python eval_cls.py --rot 45 --i $CLS_I --output_dir images --exp_name rot45
python eval_seg.py --rot 45 --i $SEG_I --output_dir images --exp_name rot45

echo "== Q3 number of points (points, cls acc, seg acc) =="
for n in 10000 5000 1000 500 100 50 10; do
    echo "$n $(python eval_cls.py --num_points $n --output_dir output | acc) $(python eval_seg.py --num_points $n --output_dir output | acc)"
done
python eval_cls.py --num_points 100 --i $CLS_I --output_dir images --exp_name pts100
python eval_seg.py --num_points 100 --i $SEG_I --output_dir images --exp_name pts100

echo "== Q4 PointNet++ =="
python eval_cls.py --pp --i $CLS_I --output_dir images --exp_name pp
python eval_seg.py --pp --i $SEG_I --output_dir images --exp_name pp
} 2>&1 | grep -v "successfully loaded" | tee output/results.txt
