import typing
import datetime
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from io import BytesIO

from src.data.make_dataset import load_debit
from src.data.make_credit_dataset import load_credit
from src.models import train_dataset_predict, credit_dataset_predict


router = APIRouter()


class PredictData(BaseModel):
    original: typing.Dict[datetime.date, float]
    predict: typing.Dict[datetime.date, float]


class PredictResponse(BaseModel):
    credit: typing.Optional[PredictData]
    debit: typing.Optional[PredictData]



@router.post('/predict', response_model=PredictResponse)
async def predict_file(file: UploadFile):
    
    data = PredictResponse()
    
    file_content = BytesIO(file.file.read())
    
    debit_original_data = {}
    debit_predict_data = {}
                    
    debit_df = load_debit(file_content)
    debif_pred = train_dataset_predict(file_content)
        
    for dt, val in debit_df.to_dict()['Debit'].items():
        debit_original_data[dt.date()] = val
        
    for dt, val in debif_pred.to_dict().items():
        debit_predict_data[dt.date()] = val
        
    data.debit = PredictData(original=debit_original_data, predict=debit_predict_data)
        
    credit_original_data = {}
    credit_predict_data = {}
        
    credit_df = load_credit(file_content)
    credit_pred = credit_dataset_predict(file_content)
        
    for dt, val in credit_df.to_dict()['Credit'].items():
        credit_original_data[dt.date()] = val
        
    for dt, val in credit_pred.to_dict().items():
        credit_predict_data[dt.date()] = val
        
    data.credit = PredictData(original=credit_original_data, predict=credit_predict_data)
    
    return data