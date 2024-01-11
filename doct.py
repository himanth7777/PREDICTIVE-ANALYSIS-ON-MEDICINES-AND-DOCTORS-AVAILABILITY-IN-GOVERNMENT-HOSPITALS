


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report





data = {
    'Time': ['10:00', '13:30', '15:45', '16:30', '11:15', '14:00', '16:30', '08:45', '12:30', '15:00', '09:15', '11:45', '14:15', '17:00'],
    'Date': ['12/04/2022', '17/04/2022', '22/04/2022', '27/04/2022', '03/05/2022', '08/05/2022', '13/05/2022', '18/05/2022', '23/05/2022', '28/05/2022', '02/06/2022', '07/06/2022', '12/06/2022', '17/06/2022'],
    'DoctorName': ['Dr. Sneha Kapoor', 'Dr. Vikas Deshmukh', 'Dr. Ayesha Patel', 'Dr. Sameera Singh', 'Dr. Varun Reddy', 'Dr. Kavita Desai', 'Dr. Rishi Verma', 'Dr. Nandini Gupta', 'Dr. Abhinav Patel', 'Dr. Ananya Kumar', 'Dr. Vivek Joshi', 'Dr. Ishita Malhotra', 'Dr. Prateek Deshmukh', 'Dr. Sanjana Yadav'],
    'Specialization': ['Cardiology', 'Orthopedics', 'Dermatology', 'Pediatrics', 'Gynecology', 'Cardiology', 'Orthopedics', 'Dermatology', 'Pediatrics', 'Gynecology', 'Cardiology', 'Orthopedics', 'Dermatology', 'Pediatrics'],
    'AppointmentHistory': ['Moderate', 'High', 'Low', 'Moderate', 'High', 'Low', 'Moderate', 'High', 'Low', 'Moderate', 'High', 'Low', 'High', 'Moderate'],
    'VacationLeaveInfo': ['None', '15-20 Apr 2022', 'None', 'None', 'None', '08-15 May 2022', 'None', 'None', '23-28 May 2022', 'None', 'None', 'None', 'None', 'None'],
    'Location': ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore', 'Kanpur', 'Ahmedabad', 'Pune', 'Delhi', 'Mumbai', 'Chennai', 'Pune', 'Ahmedabad', 'Kanpur'],
    'PatientLoad': [20, 15, 25, 18, 22, 30, 16, 12, 28, 21, 19, 24, 14, 17],
    'AvailabilityStatus': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No']
}





from sklearn.preprocessing import LabelEncoder




label_encoder = LabelEncoder()


data['Time'] = label_encoder.fit_transform(data['Time'])
data['Date'] = label_encoder.fit_transform(data['Date'])
data['DoctorName'] = label_encoder.fit_transform(data['DoctorName'])
data['Specialization'] = label_encoder.fit_transform(data['Specialization'])
data['AppointmentHistory'] = label_encoder.fit_transform(data['AppointmentHistory'])
data['VacationLeaveInfo'] = label_encoder.fit_transform(data['VacationLeaveInfo'])
data['Location'] = label_encoder.fit_transform(data['Location'])
data['AvailabilityStatus'] = label_encoder.fit_transform(data['AvailabilityStatus'])


print(data)
from sklearn.preprocessing import LabelEncoder




label_encoder = LabelEncoder()

df = pd.DataFrame(data)
# Print the updated data
print(data)






X = df[['Specialization', 'AppointmentHistory','PatientLoad']]
y = df['AvailabilityStatus']






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





rf_model = RandomForestClassifier(n_estimators=100, random_state=42)





rf_pred=rf_model.fit(X_train, y_train)





y_pred = rf_model.predict(X_test)





accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}')




import pickle





pickle.dump(rf_pred,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))







