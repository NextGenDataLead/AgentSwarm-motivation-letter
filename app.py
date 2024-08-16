from flask import Flask, request, render_template, send_file
import pandas as pd
from Motivation_letter_flask import generate_motivation_letter
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = None
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
    job_description = request.form['job_description']

    if file:
        df = pd.read_excel(file)
    elif 'resumeCached' in request.cookies:
        # Load from cache
        df = pd.read_excel(BytesIO(request.cookies.get('resumeCached')))
    else:
        return "No file uploaded and no cached resume available", 400

    # Generate the motivation letter
    motivation_letter_io = generate_motivation_letter(df, job_description)

    # Send the file for download
    return send_file(motivation_letter_io, as_attachment=True, download_name="motivation_letter.docx")

@app.route('/download-template')
def download_template():
    # Path to the template file
    template_path = os.path.join('templates', 'Resume_Template.xlsx')
    return send_file(template_path, as_attachment=True, download_name="Resume_Template.xlsx")

if __name__ == '__main__':
    app.run(debug=True)
