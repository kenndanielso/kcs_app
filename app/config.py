import os

WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

TEMP_PLOT_DIR = 'app/static/temp'
if not os.path.exists(TEMP_PLOT_DIR):
	os.makedirs(TEMP_PLOT_DIR)