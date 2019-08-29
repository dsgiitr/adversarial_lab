
from flask import Flask, render_template, request 
  
# Flask constructor takes the name of  
# current module (__name__) as argument. 
app = Flask(__name__, template_folder='template') 
  
# The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function. 
@app.route('/sakshi') 
# ‘/’ URL is bound with hello_world() function. 
# def hello_world(): 
#     return 'Welcome'
def welcomemessage():
	return 'Fooling the Neural Network with Adversarial Attack'

@app.route('/upload')
def uploadfile():
   return render_template('upload.html')


	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		return 'file uploaded successfully'




# main driver function 
if __name__ == '__main__': 
  
    # run() method of Flask class runs the application  
    # on the local development server. 
    app.run() 