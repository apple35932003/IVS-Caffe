#python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_10000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_10k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_20000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_20k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_30000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_30k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_40000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_40k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_50000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_50k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_60000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_60k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_70000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_70k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_80000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_80k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_90000.caffemodel  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_90k.log
python -u ./tools/test_net.py --gpu 3 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC10/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC10_iter_100000.caffemodel --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC10_result_100k.log