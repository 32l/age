import caffe

fn_model = 'external/AgeEstimation0630/age_model/fix_weight/deploy.prototxt'
fn_weight = 'external/AgeEstimation0630/age_model/fix_weight/fix_weight_fold_0.caffemodel'

model = caffe.Net(fn_model, fn_weight, caffe.TEST)
