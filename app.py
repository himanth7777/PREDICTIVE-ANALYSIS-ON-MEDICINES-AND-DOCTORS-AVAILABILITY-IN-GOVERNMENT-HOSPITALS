import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))


def predict_forest(Specialization, AppointmentHistory, Location):
    input = np.array([[Specialization, AppointmentHistory, Location]]).astype(np.float64)
    prediction = model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)


def main():
    st.title("Doctor Availability")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Doctor Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    specialization_options = ["1. Orthopedic", "2. Cardiologist", "3. Dermatologist"]
    specialization_serial_number = st.selectbox("Specialization", specialization_options)


    specialization = specialization_serial_number.split(".")[0].strip()

    appointment = st.text_input("Appointment", "Type Here")

    location_options = ["1. Bangalore", "2. Chennai", "3. Kanpur"]
    location_serial_number = st.selectbox("Location", location_options)


    location = location_serial_number.split(".")[0].strip()

    safe_html = """  
      <div style="background-color:#F4D03F;padding:10px >
        <h2 style="color:white;text-align:center;"> The Doctor is Not Available</h2>
      </div>
    """
    danger_html = """  
      <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black;text-align:center;"> The Doctor is Available</h2>
      </div>
    """

    legend_html = """
    <div style="padding:10px;text-align:center;">
        <p style="font-size:16px;color:#2E86C1;">Legend:</p>
        <p style="font-size:14px;color:#F4D03F;">Not Available (Probability < 0.4)</p>
        <p style="font-size:14px;color:#F08080;">Available (Probability >= 0.4)</p>
    </div>
    """

    if st.button("Predict"):
        output = predict_forest(specialization, appointment, location)
        st.success('The probability of doctor availability is {}'.format(output))

        if output > 0.4:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

        st.markdown(legend_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
