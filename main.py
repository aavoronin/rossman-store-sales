from check_GPU import check_GPU

check_GPU()

from predict_rossman_sales import predict_rossman_sales

if __name__ == '__main__' :
    p = predict_rossman_sales()
    p.load_data()
    p.design_features()
    p.init_list_of_models()
    p.run_models()
    p.done()
