import os
import base64

import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import black
import fitz # PyMuPDF

from mlel.ratings import aggregate_rating


C_SIZE = (1560, 2411)
POS_TEXT = {
    "name":                                     ('drawString',        90, '-Bold', .04,  .855, None),
    "result_type":                              ('drawString',        90, '',      .04,  .815, None),
    "environment":                              ('drawString',        90, '',      .04,  .42,  None),
    "dataset":                                  ('drawRightString',   90, '',      .95,  .815, None),
    "inference_power_draw":                     ('drawRightString',   68, '-Bold', .25,  .25,  None),
    "parameters":                               ('drawRightString',   68, '-Bold', .69,  .05,  None),
    "inference_time":                           ('drawRightString',   68, '-Bold', .75,  .25,  None),
    "train_time":                               ('drawRightString',   68, '-Bold', .7,  .25,  '{} /'),
    "train_time_epoch":                         ('drawString',        68, '-Bold', .715, .25,  None),
    "train_power_draw":                         ('drawRightString',   68, '-Bold', .2,  .25,  '{} /'),
    "train_power_draw_epoch":                   ('drawString',        68, '-Bold', .215,  .25,  None),
    "top1_val":                                 ('drawRightString',   68, '-Bold', .22,  .05,  '{} /'),
    "top5_val":                                 ('drawString',        68, '-Bold', .235, .05,  None),
    # TODO add field for power draw sources
    "$Inference Power Draw per Sample":         ('drawCentredString', 56, '',      .25,  .22,  None),
    "$Inference Runtime per Sample":            ('drawCentredString', 56, '',      .75,  .22,  None),
    "$Inference Top-1 / Top-5 Accuracy":        ('drawCentredString', 56, '',      .25,  .02,  None),
    "$Inference Model Size":                    ('drawCentredString', 56, '',      .75,  .02,  None),
    "$Inference Ws":                            ('drawString',        56, '',      .27,  .25,  None),
    "$Inference ms":                            ('drawString',        56, '',      .77,  .25,  None),
    "$Inference [%]":                           ('drawString',        56, '',      .34,  .05,  None),
    "$Inference M Parameters":                  ('drawString',        56, '',      .7,   .05,  None),
    "$Training Power Draw Total / per Sample":  ('drawCentredString', 56, '',      .25,  .22,  None),
    "$Training Runtime Total / per Sample":     ('drawCentredString', 56, '',      .75,  .22,  None),
    "$Training Top-1 / Top-5 Accuracy":         ('drawCentredString', 56, '',      .25,  .02,  None),
    "$Training Model Size":                     ('drawCentredString', 56, '',      .75,  .02,  None),
    "$Training kWh":                            ('drawString',        56, '',      .31,  .25,  None),
    "$Training h":                              ('drawString',        56, '',      .81,  .25,  None),
    "$Training [%]":                            ('drawString',        56, '',      .34,  .05,  None),
    "$Training M Parameters":                   ('drawString',        56, '',      .7,   .05,  None),
}
POS_RATINGS = { char: (.66, y) for char, y in zip('ABCDE', reversed(np.linspace(.461, .727, 5))) }
ICON_NAME_TO_METRIC = {
    'Inference': {
        'time': 'inference_time',
        'top': 'top1_val',
        'power_draw': 'inference_power_draw',
        'parameters': 'parameters'
    },
    'Training': {
        'time': 'train_time',
        'top': 'top1_val',
        'power_draw': 'train_power_draw',
        'parameters': 'parameters'
    }
}
ICON_POS = {
    'time': (1050, 680),
    'top': (250, 200),
    'power_draw': (250, 770),
    'parameters': (1050, 200)
}


def summary_to_label(summary, rating_mode):
    canvas = Canvas("result.pdf", pagesize=C_SIZE)
    
    frate = aggregate_rating(summary, rating_mode, 'ABCDE')
    canvas.drawInlineImage(os.path.join("label_design", "parts", f"bg.png"), 0, 0)
    for icon, (posx, posy) in ICON_POS.items():
        metric = ICON_NAME_TO_METRIC[summary['result_type']][icon]
        rating = summary[metric]['rating']
        canvas.drawInlineImage(os.path.join("label_design", "parts", f"{icon}_{rating}.png"), posx, posy)
    canvas.drawInlineImage(os.path.join("label_design", "parts", f"Rating_{frate}.png"), POS_RATINGS[frate][0] * C_SIZE[0], POS_RATINGS[frate][1] * C_SIZE[1])
    canvas.setFillColor(black)
    canvas.setLineWidth(3) # add stroke to make even bigger letters
    canvas.setStrokeColor(black)
    text=canvas.beginText()
    text.setTextRenderMode(2)
    canvas._code.append(text.getCode())
    # draw text parts
    for key, (draw_method, fsize, style, x, y, fmt) in POS_TEXT.items():
        draw_method = getattr(canvas, draw_method)
        canvas.setFont('Helvetica' + style, fsize)
        if key.startswith(f"${summary['result_type']} "):
            # static text on label depending on the result type
            text = key.replace(f"${summary['result_type']} ", "")
        elif key in summary:
            # dynamic text that receives content from summary
            text = summary[key] if isinstance(summary[key], str) else f'{summary[key]["value"]:4.2f}'[:4]
        else:
            text = None
        if text is not None:
            if fmt is not None:
                text = fmt.format(text)
            draw_method(int(C_SIZE[0] * x), int(C_SIZE[1] * y), text)
    pdf_doc = fitz.Document(stream=canvas.getpdfdata(), filetype='pdf')
    label_bytes = pdf_doc.load_page(0).get_pixmap().tobytes()
    base64_enc = base64.b64encode(label_bytes).decode('ascii')
    return 'data:image/png;base64,{}'.format(base64_enc), pdf_doc

if __name__ == "__main__":
    test_inf = {'environment': 'A100 x8 - TensorFlow 2.6.2', 'name': 'MobileNetV3Large', 'dataset': 'ImageNet', 'result_type': 'Inference', 'parameters': {'value': 5.507432, 'index': 8.117608351769027, 'rating': 0}, 'fsize': {'value': 22733832, 'index': 7.903654782000676, 'rating': 0}, 'inference_power_draw': {'value': 0.19723524031291068, 'index': 1.8633597887444289, 'rating': 0}, 'inference_time': {'value': 0.3874090093521635, 'index': 1.094596240474881, 'rating': 1}, 'top1_val': {'value': 0.7107800245285034, 'index': 0.9904961488720718, 'rating': 2}, 'top5_val': {'value': 0.8927800059318542, 'index': 0.9963840212294235, 'rating': 2}}
    test_train = {'environment': 'A100 x8 - TensorFlow 2.6.2', 'name': 'MobileNetV3Small', 'dataset': 'ImageNet', 'result_type': 'Training', 'parameters': {'value': 2.5549679999999997, 'index': 17.498135397390495, 'rating': 0}, 'fsize': {'value': 10804008, 'index': 16.630898459164413, 'rating': 0}, 'train_power_draw_epoch': {'value': 0.0938475629776553, 'index': 3.263693961714361, 'rating': 0}, 'train_power_draw': {'value': 56.30853778659318, 'index': 0.4895540942571542, 'rating': 4}, 'train_time_epoch': {'value': 0.12829326028029125, 'index': 1.6842725959307598, 'rating': 0}, 'train_time': {'value': 76.97595616817475, 'index': 0.25264088938961404, 'rating': 4}, 'top1_val': {'value': 0.6313999891281128, 'index': 0.8798773685911093, 'rating': 4}, 'top5_val': {'value': 0.8371400237083435, 'index': 0.9342872125412293, 'rating': 4}}
    for idx, summary in enumerate([test_inf, test_train]):
        _, pdf_doc = summary_to_label(summary, 'mean')
        pdf_doc.save(f'testlabel_{idx}.pdf')