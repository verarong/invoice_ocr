from fastapi import FastAPI
from app.utils import debug
from app.handler import OcrEngine, east_client, ocr_client, angle_client, maskrcnn_client, TxOutputParser, \
    SegmentationOutputParser, MultiOutputParser
from app.items import InputItem, SegInputItem

app = FastAPI()


@app.get("/health")
def read_root():
    return {"project": "alive"}


@debug
@app.post('/predict')
def predict(item: InputItem):
    engine = OcrEngine(east_client, ocr_client, angle_client, item.InvoiceType)
    predicts = engine.predict(item)
    output_parser = TxOutputParser(item, *predicts)
    return output_parser.parse_output(item.InvoiceType)


@debug
@app.post('/segmentation')
def segmentation(item: SegInputItem):
    engine = OcrEngine(east_client, ocr_client, angle_client, maskrcnn=maskrcnn_client)
    predicts = engine.segmentation_predict(item)
    output_parser = SegmentationOutputParser(item, predicts[2])

    return output_parser.parse_output()


@debug
@app.post('/multi_invoices_predict')
def multi_invoices_predict(item: SegInputItem):
    engine = OcrEngine(east_client, ocr_client, angle_client, maskrcnn=maskrcnn_client)

    predicts = engine.multi_invoices_predict(item)
    output_parser = MultiOutputParser(item, *predicts)

    # print("nicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenicenice")

    return output_parser.parse_output()
