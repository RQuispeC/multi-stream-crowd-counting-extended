run_ucf =  python3 main.py -d ucf-cc-50
run_shanghai = python3 main.py -d shanghai-tech

# UCF-CC-50 DATASET
# augment data of ucf-cc-50, run this just once
augment-ucf: augment-ucf-same augment-ucf-knn augment-ucf-face
augment-ucf-same:
	$(run_ucf) --augment-only --force-augment --force-den-maps --gt-mode same
augment-ucf-knn:
	$(run_ucf) --augment-only --force-augment --force-den-maps --gt-mode knn
augment-ucf-face:
	#python3 tiny_detection_mxnet.py -d ucf-cc-50 --save-plots --root-dir /workspace/quispe
	$(run_ucf) --augment-only --force-augment --force-den-maps --gt-mode face

train-ucf-mcnn1-same:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn1 --gt-mode same --model mcnn1
test-ucf-mcnn1-same: 
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn1/ --resume log/mcnn1/ucf-cc-50_people_thr_0_gt_mode_same/ --gt-mode same --model mcnn1

train-ucf-mcnn1-knn:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn1 --gt-mode knn --model mcnn1
test-ucf-mcnn1-knn:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn1/ --resume log/mcnn1/ucf-cc-50_people_thr_0_gt_mode_knn/ --gt-mode knn --model mcnn1

train-ucf-mcnn1-face:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn1 --gt-mode face --model mcnn1
test-ucf-mcnn1-face: 
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn1/ --resume log/mcnn1/ucf-cc-50_people_thr_0_gt_mode_face/ --gt-mode face --model mcnn1

train-ucf-mcnn2-same:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn2 --gt-mode same --model mcnn2
test-ucf-mcnn2-same:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn2/ --resume log/mcnn2/ucf-cc-50_people_thr_0_gt_mode_same/ --gt-mode same --model mcnn2

train-ucf-mcnn2-knn:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn2 --gt-mode knn --model mcnn2
test-ucf-mcnn2-knn:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn2/ --resume log/mcnn2/ucf-cc-50_people_thr_0_gt_mode_knn/ --gt-mode knn --model mcnn2

train-ucf-mcnn2-face:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn2 --gt-mode face --model mcnn2
test-ucf-mcnn2-face:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn2/ --resume log/mcnn2/ucf-cc-50_people_thr_0_gt_mode_face/ --gt-mode face --model mcnn2

train-ucf-mcnn3-same:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn3 --gt-mode same --model mcnn3
test-ucf-mcnn3-same:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn3/ --resume log/mcnn3/ucf-cc-50_people_thr_0_gt_mode_same/ --gt-mode same --model mcnn3

train-ucf-mcnn3-knn:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn3 --gt-mode knn --model mcnn3
test-ucf-mcnn3-knn: 
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn3/ --resume log/mcnn3/ucf-cc-50_people_thr_0_gt_mode_knn/ --gt-mode knn --model mcnn3

train-ucf-mcnn3-face:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn3 --gt-mode face --model mcnn3
test-ucf-mcnn3-face:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn3/ --resume log/mcnn3/ucf-cc-50_people_thr_0_gt_mode_face/ --gt-mode face --model mcnn3

train-ucf-mcnn4-same:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn4 --gt-mode same --model mcnn4
test-ucf-mcnn4-same:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn4/ --resume log/mcnn4/ucf-cc-50_people_thr_0_gt_mode_same/ --gt-mode same --model mcnn4

train-ucf-mcnn4-knn:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn4 --gt-mode knn --model mcnn4
test-ucf-mcnn4-knn:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn4/ --resume log/mcnn4/ucf-cc-50_people_thr_0_gt_mode_knn/ --gt-mode knn --model mcnn4

train-ucf-mcnn4-face:
	$(run_ucf) --train-batch 32 --save-dir log/mcnn4 --gt-mode face --model mcnn4
test-ucf-mcnn4-face:
	$(run_ucf) --evaluate-only --save-plots --save-dir log/mcnn4/ --resume log/mcnn4/ucf-cc-50_people_thr_0_gt_mode_face/ --gt-mode face --model mcnn4

# SHANGHAI-TECH DATASET
# augment data of Shanghai Tech, do this just once
augment-shanghai: augment-shanghai-same augment-shanghai-knn augment-shanghai-face
augment-shanghai-same:
	$(run_shanghai) --augment-only --force-augment --force-den-maps --gt-mode same
augment-shanghai-knn:
	$(run_shanghai) --augment-only --force-augment --force-den-maps --gt-mode knn
augment-shanghai-face:
	#python3 tiny_detection_mxnet.py -d shanghai-tech --save-plots --root-dir /workspace/quispe
	$(run_shanghai) --augment-only --force-augment --force-den-maps --gt-mode face

train-shanghai-mcnn1-same:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn1 --gt-mode same --model mcnn1
test-shanghai-mcnn1-same:
	$(run_shanghai) --save-dir log/mcnn1 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots --gt-mode same --model mcnn1

train-shanghai-mcnn1-knn:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn1 --gt-mode knn --model mcnn1
test-shanghai-mcnn1-knn:
	$(run_shanghai) --save-dir log/mcnn1 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_knn/ --evaluate-only --save-plots --gt-mode knn --model mcnn1

train-shanghai-mcnn1-face:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn1 --gt-mode face --model mcnn1
test-shanghai-mcnn1-face:
	$(run_shanghai) --save-dir log/mcnn1 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_face/ --evaluate-only --save-plots --gt-mode face --model mcnn1

train-shanghai-mcnn2-same:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn2 --gt-mode same --model mcnn2
test-shanghai-mcnn2-same:
	$(run_shanghai) --save-dir log/mcnn2 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots --gt-mode same --model mcnn2

train-shanghai-mcnn2-knn:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn2 --gt-mode knn --model mcnn2
test-shanghai-mcnn2-knn:
	$(run_shanghai) --save-dir log/mcnn2 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_knn/ --evaluate-only --save-plots --gt-mode knn --model mcnn2

train-shanghai-mcnn2-face:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn2 --gt-mode face --model mcnn2
test-shanghai-mcnn2-face:
	$(run_shanghai) --save-dir log/mcnn2 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_face/ --evaluate-only --save-plots --gt-mode face --model mcnn2

train-shanghai-mcnn3-same:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn3 --gt-mode same --model mcnn3
test-shanghai-mcnn3-same:
	$(run_shanghai) --save-dir log/mcnn3 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots --gt-mode same --model mcnn3

train-shanghai-mcnn3-knn:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn3 --gt-mode knn --model mcnn3
test-shanghai-mcnn3-knn:
	$(run_shanghai) --save-dir log/mcnn3 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_knn/ --evaluate-only --save-plots --gt-mode knn --model mcnn3

train-shanghai-mcnn3-face:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn3 --gt-mode face --model mcnn3
test-shanghai-mcnn3-face:
	$(run_shanghai) --save-dir log/mcnn3 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_face/ --evaluate-only --save-plots --gt-mode face --model mcnn3

train-shanghai-mcnn4-same:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn4 --gt-mode same --model mcnn4
test-shanghai-mcnn4-same:
	$(run_shanghai) --save-dir log/mcnn4 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots --gt-mode same --model mcnn4

train-shanghai-mcnn4-knn:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn4 --gt-mode knn --model mcnn4
test-shanghai-mcnn4-knn:
	$(run_shanghai) --save-dir log/mcnn4 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_knn/ --evaluate-only --save-plots --gt-mode knn --model mcnn4

train-shanghai-mcnn4-face:
	$(run_shanghai)  --train-batch 32 --save-dir log/mcnn4 --gt-mode face --model mcnn4
test-shanghai-mcnn4-face:
	$(run_shanghai) --save-dir log/mcnn4 --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_face/ --evaluate-only --save-plots --gt-mode face --model mcnn4

debug:
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn1
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn1
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn2
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn2
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn3
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn3
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn4
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode same --max-epoch 1 --model mcnn4

	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn1
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn1
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn2
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn2
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn3
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn3
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn4
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode knn --max-epoch 1 --model mcnn4

	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn1
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn1
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn2
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn2
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn3
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn3
	$(run_ucf) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn4
	$(run_shanghai) --train-batch 32 --save-dir log/tmp --gt-mode face --max-epoch 1 --model mcnn4