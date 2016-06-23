## App modules
from flask import Blueprint, request, render_template
import tempfile
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required


## Python modules
import numpy as numpy
import pandas as pandas
import seaborn as sns
from . import app
from . import summarizer
from .config import TEMP_PLOT_DIR

class DisplayForm(Form):
	text = fields.TextAreaField('Text:',validators=[Required()])
	submit = fields.SubmitField('Submit')

@app.route('/', methods=["GET", "POST"])
def index():
	# need to assign input for business
	form = DisplayForm()
	summary = None

	if form.validate_on_submit():
		# store the submitted values
		submitted_data = form.data
		print(submitted_data)

		# Retrieve values from form
		text = submitted_data['text']

		# Return summary
		summary = summarizer.textrank_summarizer(text)

	return render_template('index.html',form=form,summary=summary) 

