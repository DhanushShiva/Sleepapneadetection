from flask import Flask, render_template, request, flash, redirect,url_for
from flask import Flask, render_template_string
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import smtplib
from email.message import EmailMessage
import csv
import os
import bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def predict(values):
    if len(values) == 11:
        data = pd.read_csv('preprocessed_datset.csv')

        # Split the data into features and target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the input data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Load the model
        loaded_model = keras.models.load_model('ann_3.h5')

        # Evaluate the loaded model on the test set
        test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

        # Use the loaded model to make predictions on new data
        new_data = np.array([values])
        new_data_scaled = scaler.transform(new_data)
        prediction = loaded_model.predict(new_data_scaled)
        print(prediction)

        predicted_class = np.argmax(prediction[0])
        print(f'Predicted class: {predicted_class}')
        return predicted_class
    else:
        print("Check the number of features entered")
        return None

def send_email(subject, body, from_email, to_emails, smtp_server, smtp_port, smtp_username, smtp_password, bcc_email):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_emails
    msg['Bcc'] = ','.join(bcc_email)
    msg.set_content(body)
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.send_message(msg)
    server.quit()
    print("Email sent successfully!")

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/docter')
def docter():
    df = pd.read_csv("Log.csv")
    table_html = df.to_html(classes='table table-striped', index=False)
    return render_template('docter.html', table_html=table_html)
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']

        # Establish connection to SQLite database
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        # Create table if it doesn't exist
        cursor.execute("CREATE TABLE IF NOT EXISTS user (name TEXT, password TEXT, mobile TEXT, email TEXT)")

        # Check if the user already exists
        cursor.execute("SELECT * FROM user WHERE name=?", (name,))
        existing_user = cursor.fetchone()

        if existing_user:
            connection.close()
            return render_template('index.html', msg='User already exists')

        # Insert new user into the table
        cursor.execute("INSERT INTO user (name, password, mobile, email) VALUES (?, ?, ?, ?)",
                       (name, password, mobile, email))
        connection.commit()
        connection.close()

        return render_template('index.html', msg='Successfully registered')

    return render_template('index.html')

@app.route('/docterreg', methods=['GET', 'POST'])
def docterreg():
    if request.method == 'POST':
        secretkey = request.form['secretkey']
        if secretkey == 'mindyourownbussiness':  # Replace with your actual secret key
            try:
                # Establish connection to SQLite database
                connection = sqlite3.connect('docter_data.db')
                cursor = connection.cursor()

                # Retrieve doctor's username and password from the form
                docterusername = request.form['docterusername']
                docterpassword = request.form['docterpassword']

                # Create the table if it doesn't exist
                cursor.execute("""CREATE TABLE IF NOT EXISTS docter (
                                  docterusername TEXT,
                                  docterpassword TEXT
                                  )""")

                # Insert new doctor data into the table
                cursor.execute("INSERT INTO docter (docterusername, docterpassword) VALUES (?, ?)",
                               (docterusername, docterpassword))

                # Commit changes and close connection
                connection.commit()
                connection.close()

                # Redirect to login page with success message
                return render_template('docterlog.html', msg='Successfully registered')

            except sqlite3.Error as e:
                # Handle database errors
                print(f"Database error: {e}")
                error_msg = 'Database error. Please try again later.'
                return render_template('docterreg.html', error=error_msg)

        else:
            # Render registration page with error message for incorrect secret key
            error_msg = 'Incorrect secret key. Please try again.'
            return render_template('docterreg.html', error=error_msg)

    # Render the registration page if it's a GET request
    return render_template('docterreg.html')
@app.route('/docterlog', methods=['GET', 'POST'])
def docterlog():
    if request.method == 'POST':
        # Retrieve username and password from the form
        docterusername = request.form['username']
        docterpassword = request.form['password']

        # Establish connection to SQLite database
        connection = sqlite3.connect('docter_data.db')
        cursor = connection.cursor()

        try:
            # Query to check if the doctor's credentials exist in the database
            query = "SELECT * FROM docter WHERE docterusername = ? AND docterpassword = ?"
            cursor.execute(query, (docterusername, docterpassword))

            # Fetch the result
            result = cursor.fetchone()

            if result:
                # Redirect to doctor's dashboard if credentials are correct
                return redirect(url_for('docter'))
            else:
                # Render login page with error message if credentials are incorrect
                error_msg = 'Incorrect credentials. Please try again.'
                return render_template('docterlog.html', error=error_msg)

        except sqlite3.Error as e:
            # Handle database errors
            print(f"Database error: {e}")
            error_msg = 'Database error. Please try again later.'
            return render_template('docterlog.html', error=error_msg)

        finally:
            # Close cursor and connection
            cursor.close()
            connection.close()

    # Render the login page if it's a GET request
    return render_template('docterlog.html')

# Route for doctor's dashboard
@app.route('/docter_dashboard')
def docter_dashboard():
    return render_template('docter_dashboard.html')  # Replace with your actual dashboard template

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '" + name + "' AND password= '" + password + "'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('kidney.html')

    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route("/Apnea", methods=['GET', 'POST'])
def Apnea():
    return render_template('kidney.html')

@app.route('/stage1')
def stage1():
    return render_template('stage1.html')

@app.route('/stage2')
def stage2():
    return render_template('stage2.html')

@app.route('/stage3')
def stage3():
    return render_template('stage3.html')

@app.route('/stage4')
def stage4():
    return render_template('stage4.html')


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        name = list(to_predict_dict.values())[0]
        to_predict_list = list(map(float, list(to_predict_dict.values())[1:]))
        print("Prediction list is {}".format(to_predict_list))

        result = predict(to_predict_list)

        # Store data in CSV file
        if result is not None:
            s_data = [name] + to_predict_list + [result]
            csv_file = 'Log.csv'

            file_exists = os.path.isfile(csv_file)
            columns = ['Name', 'BQ', 'ESS', 'BMI', 'Weight', 'Height', 'Head', 'Neck', 'Waist', 'Buttock', 'Age',
                       'Gender', 'Result']

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(columns)
                writer.writerow(s_data)
            print("Data stored successfully in CSV file.")

            if result == 0:
                res = "NORMAL"

            elif result == 1:
                res = "MILD APNEA"

            elif result == 2:
                res = "MODERATE APNEA"

            elif result == 3:
                res = "SEVERE APNEA"

            connection = sqlite3.connect('user_data.db')
            cursor = connection.cursor()
            cursor.execute("SELECT email FROM user WHERE name = ?", (name,))
            email = cursor.fetchone()[0]  # Fetch the first email found
            subject = "Sleep Apnea Prediction Result"
            from_email = "sleepapneaprediction1@hotmail.com"
            to_emails = [email]  # Ensure it's a list with the email address
            bcc_email = ["dhanushs@bgsit.ac.in"]
            smtp_server = "smtp-mail.outlook.com"
            smtp_port = 587
            smtp_username = "sleepapneaprediction1@hotmail.com"
            smtp_password = "sleepapnea@2024"  # Replace with your actual SMTP password

            body = f"Hello {name},\n\nYour sleep apnea prediction result is: {res}.\n\nBest regards,\nSleep Apnea Prediction Team"

            send_email(subject, body, from_email, to_emails, smtp_server, smtp_port, smtp_username, smtp_password,bcc_email)
            print(f"Stage is {res}")

            return render_template('predict.html', pred=result, name=name, res=res)
        else:
            return render_template('predict.html', pred=None, name=name, res="Invalid input length")

    return render_template('predict.html', pred=None, name='')


if __name__ == '__main__':
    app.run(debug=True)
