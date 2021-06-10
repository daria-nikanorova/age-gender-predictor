import logging
import cv2 as cv
import numpy as np
from io import BytesIO

from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.utils.executor import start_webhook

from bot.settings import (BOT_TOKEN, HEROKU_APP_NAME,
                      WEBHOOK_URL, WEBHOOK_PATH,
                      WEBAPP_HOST, WEBAPP_PORT)

from bot.settings import (FACE_PROTO, FACE_MODEL,
                      AGE_PROTO, AGE_MODEL,
                      GENDER_PROTO, GENDER_MODEL,
                      MEAN_VALUES, AGE_LIST,
                      GENDER_LIST)

bot = Bot(BOT_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Hi!\nDo you want to predict someone's age and gender by photo?")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Hi!\nSend me a photo and I'll try to guess your age and gender!")


@dp.message_handler()
async def echo_message(message: types.Message):
    await message.reply("Hi!\nSend me a photo and I'll try to guess your age and gender!")


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message: types.Message):
    logging.warning(f'Recieved a message from {message.from_user}')
    downloaded = await bot.download_file_by_id(message.photo[-1].file_id)
    b = BytesIO()
    b.write(downloaded.getvalue())
    b.seek(0)
    image = cv.imdecode(np.fromstring(b.read(), np.uint8), 1)

    faceNet, ageNet, genderNet = fit_models(FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL)

    output_image, output_message = age_gender_predict(image, faceNet, ageNet, genderNet, AGE_LIST, GENDER_LIST, MEAN_VALUES)

    if output_image is not None:
        image = cv.imencode('test.jpg', output_image)
        b.seek(0)
        await bot.send_photo(message.chat.id, image[1].tobytes(), caption=output_message)


def get_face(net, frame, conf_threshold=0.7):
    frame_for_box = frame.copy()
    frame_height = frame_for_box.shape[0]
    frame_width = frame_for_box.shape[1]
    blob = cv.dnn.blobFromImage(frame_for_box, 1.0, (300, 300), [104, 117, 123])

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frame_width)
            y1 = int(detections[0, 0, i, 4]*frame_height)
            x2 = int(detections[0, 0, i, 5]*frame_width)
            y2 = int(detections[0, 0, i, 6]*frame_height)
            face_boxes.append([x1, y1, x2, y2])

    return face_boxes, frame_height


def age_gender_predict(frame, face_net, age_net, gender_net, age_lst, gender_lst, model_mean_values):
    boxes, height = get_face(face_net, frame)
    final_text = ''
    if not boxes:
        final_text = "No face detected"

    all_result_text = []
    for bbox in boxes:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if len(face) > 0:
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_lst[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            age = age_lst[i]
            ageConfidence = round((age_preds[0][i] * 100), 2)

            result_text = f'Gender: {gender} \n'\
                          f'Age: {age} years, {ageConfidence}%'
            all_result_text.append(result_text)

            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), int(round(height/150)), 8)
            cv.putText(frame, f'{gender}, {age}', (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

            final_text = '\n'.join(all_result_text)
    return frame, final_text


def fit_models(faceProto, faceModel, ageProto, ageModel, genderProto, genderModel):
    face_net = cv.dnn.readNetFromCaffe(faceProto, faceModel)
    age_net = cv.dnn.readNetFromCaffe(ageProto, ageModel)
    gender_net = cv.dnn.readNetFromCaffe(genderProto, genderModel)
    return face_net, age_net, gender_net


async def on_startup(dp):
    logging.warning(
        'Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')


def main():
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )

# if __name__ == '__main__':
#     executor.start_polling(dp)
