python -u ./tools/test_net.py --gpu 1 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC8/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC8_iter_40000.caffemodel --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC8_result_40k.log
#python -u ./tools/test_net.py --gpu 1 --def ./models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC8/train_test_quant_BAC.prototxt --net ./output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC8_iter_20000.caffemodel --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb voc_2007_test 2>&1 | tee zf_q6_BAC8_result_20k.log
