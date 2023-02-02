from code.stage_2_code.Result_Loader import Result_Loader
from sklearn.metrics import classification_report


result_obj = Result_Loader()
result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'
result_obj.load()
loaded_result = result_obj.data

print(classification_report(loaded_result['true_y'], loaded_result['pred_y']))
