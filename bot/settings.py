import os


BOT_TOKEN = '1831197819:AAFGi7Q44kUBcSTC62ndWJfFQeAnn8UseTg'

# BOT_TOKEN = os.getenv('BOT_TOKEN')
# if not BOT_TOKEN:
#     print('You have forgot to set BOT_TOKEN')
#     quit()
HEROKU_APP_NAME = os.getenv('HEROKU_APP_NAME')

# webhook settings
WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

# webserver settings
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = int(os.getenv('PORT'))

# model settings
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

AGE_PROTO = "models/deploy_age.prototxt"
AGE_MODEL = "models/age_net.caffemodel"

GENDER_PROTO = "models/deploy_gender.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
