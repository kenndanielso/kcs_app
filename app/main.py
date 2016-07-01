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
from .config import TEMP_PLOT_DIR
from . import kcs_process_query

class DisplayForm(Form):
	query_terms = fields.TextField('Query Terms:',validators=[Required()])
	query_date = fields.TextField('Earliest Article Date:')
	size = fields.TextField('Number of Articles:')
	submit = fields.SubmitField('Submit')


@app.route('/', methods=["GET", "POST"])
def index():
	# need to assign input for business
	form = DisplayForm()
	table_html = None

	if form.validate_on_submit():
		# store the submitted values
		submitted_data = form.data
		print(submitted_data)

		# Retrieve values from form
		query_terms = submitted_data['query_terms']
		query_date = submitted_data['query_date']
		size = submitted_data['size']
		
    
		# Return queried df
		queried_df = kcs_process_query.query_generator(query_terms = query_terms,query_date=query_date,size=size)

		table_html = queried_df.reset_index(drop=True).to_html(classes=['u-full-width'], index=False).replace('border="1"','border="0"')

	return render_template('index.html',data_table=table_html,form=form) 

