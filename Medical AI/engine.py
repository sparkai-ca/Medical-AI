from flask import Flask, render_template, request, redirect, url_for
from utils.Tumor_detection import predict as cancer_detector
from utils.MaskRCNN import predict as cells_type_detector
from utils.Pneumonia_Classifier import predict as pneumonia_detector
import time

# flask web app instance
app = Flask(__name__)


@app.route('/')
def index():
	return(render_template('splash.html'))

@app.route('/index')
@app.route('/home')
def home():
	return(render_template('index.html'))

@app.route('/CancerDetector')
def cdetector():
	return(render_template('cdetector.html'))

@app.route('/CellsTypeDetector')
def ctdetector():
	return(render_template('ctdetector.html'))

@app.route('/PneumoniaDetector')
def pdetector():
	return(render_template('pdetector.html'))


#Api Caller for cancer detection
@app.route('/CancerDetector', methods=['POST'])
def cd():
	_image=''
	_api_response=''
	try:
		st=time.time()
		api_response = cancer_detector.predict(request.files['file'])
		_api_response = api_response[0]
		_image = api_response[1]
		et=time.time()
		print('\n',round(et-st,2),' secs\n')
	except Exception as e:
		print('\nError:  ',e,'\n')
	image = None
	if _api_response == '_/':
		image=_image
	return(render_template('cdetector.html', image=image))


#Api Caller for blood cells types detection
@app.route('/CellsTypeDetector', methods=['POST'])
def ctd():
	_image=''
	_api_response=''
	try:
		st=time.time()
		api_response = cells_type_detector.predict(request.files['file'])		
		_api_response = api_response[0]
		_image = api_response[1]
		et=time.time()
		print('\n',round(et-st,2),' secs\n')
	except Exception as e:
		print('\nError:  ',e,'\n')
	image = None
	if _api_response == '_/':
		image=_image
	return(render_template('ctdetector.html', image=image))


# Api Caller for pneumonia detection
@app.route('/PneumoniaDetector', methods=['POST'])
def pd():
	_image=''
	_api_response=''
	try:
		st=time.time()
		api_response = pneumonia_detector.predict(request.files['file'])
		_api_response = api_response[0]
		_image = api_response[1]
		et=time.time()
		print('\n',round(et-st,2),' secs\n')
	except Exception as e:
		print('\nError:  ',e,'\n')
	image = None
	info = ''
	detection = ''
	txt_clr = ''
	if _api_response == '1':
		image = _image
		detection = 'Result: Positive | Pneumonia Detected'
		info = '.'
		txt_clr = 'color : red;'
	elif _api_response == '0':
		image = _image
		detection = 'Result: Negative | Pneumonia Not Detected'
		info = 'Please Try again with an xray image of chest!'
		txt_clr = 'color : blue;'

	return(render_template('pdetector.html', detection=detection, info=info, txt_clr=txt_clr, image=image))


#------------------------------------------------------------------------------------------------------------------------------------------------
## Main

if __name__ == "__main__":
	try:
		app.run(debug=True, threaded=False)
	except Exception as e:
		print("Error: ",e)