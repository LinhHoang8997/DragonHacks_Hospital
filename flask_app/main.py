from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('test_form.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submitted', methods=['POST'])
def submitted_form():
    name = request.form['name']
    email = request.form['email']
    site = request.form['site_url']
    comments = request.form['comments']

    return render_template(
    'submitted_form.html',
    name=name,
    email=email,
    site=site,
    comments=comments)

@app.route('/result', methods=['POST'])
def result_form():
    city = request.form['city']
    state = request.form['state']
    zipcode = request.form['zipcode']
    hospitalOwnership = request.form['hospitalOwnership']

    return render_template(
    'result_form.html',
    city=city,
    state=state,
    zipcode=zipcode,
    hospitalOwnership=hospitalOwnership)

# @app.context_processor
# def override_url_for():
#     return dict(url_for=dated_url_for)
#
# def dated_url_for(endpoint, **values):
#     if endpoint == 'static':
#         filename = values.get('filename', None)
#         if filename:
#             file_path = os.path.join(app.root_path,
#                                  endpoint, filename)
#             values['q'] = int(os.stat(file_path).st_mtime)
#     return url_for(endpoint, **values)

if __name__ == "__main__":
    app.run(debug=True)
