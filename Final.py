
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import cv2
import numpy as np
import face_recognition
import os
import streamlit as st
from datetime import datetime
from datetime import date
from PIL import Image
st.set_page_config(page_title="Student Attendance System",page_icon=":pencil:",layout="wide")
flag=0

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



menu=["Home","Register new Student","Mark your attendance","Display attendance sheet"]
a=st.sidebar.selectbox("Menu",menu)

if a=="Home":
    st.title("WEL - COME TO YOUR CLASS")

    image = Image.open('img3.png')

    st.image(image)


elif a=="Register new Student":
    st.subheader("REGISTER NEW STUDENT")
    name = st.text_input('Enter your Name')
    st.write('Hello ', name)


    # image from user

    def load_image(image_file):
        img = Image.open(image_file)
        return img


    # uploaded picture

    st.subheader("UPLOAD YOUR IMAGE")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # To See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        #st.write(file_details)

        st.write("YOUR IMAGE IS INSERTED")

        # To View Uploaded Image
        st.image(load_image(image_file), width=250)

        with open(os.path.join("Training_images", image_file.name), "wb") as f:
            f.write(image_file.getbuffer())
        st.success("Image is Saved")

        lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_gsjtjybz.json")
        st_lottie(lottie_coding, height=300, key="Registered")



elif a=="Mark your attendance":
    st.subheader("MARK YOUR ATTENDANCE")
    Mark = st.button("Click here to mark your attendance  ")
    FRAME_WINDOW = st.image([])
    path = 'Training_images'
    images = []
    personName = []
    myList = os.listdir(path)


    for cu_img in myList:
        current_img = cv2.imread(f'{path}/{cu_img}')
        images.append(current_img)
        personName.append(os.path.splitext(cu_img)[0])



    def markAttendance(name):
        data = pd.read_csv('Attendance.csv')
        x = data.iloc[:, 0]
        y = x.tolist()

        if name not in y:
            print(name)
            now = datetime.now()
            today = date.today()
            dtString = now.strftime('%H:%M:%S')
            row = {'name': [name], 'date': [today], 'Time': [dtString],'status':'Present'}
            df = pd.DataFrame(row)
            data = pd.concat([df, data], ignore_index=True, axis=0)
            data.to_csv("Attendance.csv", index=False)

        else:
            aa=0

            for line in data['date']:
                if line != str(date.today()):
                    now = datetime.now()
                    today = date.today()
                    dtString = now.strftime('%H:%M:%S')
                    row = {'name': [name], 'date': [today], 'Time': [dtString],'status':'Present' }
                    df = pd.DataFrame(row)

                    if aa==0:
                        data = pd.concat([df, data], ignore_index=True, axis=0)
                        data.to_csv("Attendance.csv", index=False)
                        aa=1


    def faceEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList


    encodeListKnown = faceEncodings(images)
    print("All Encodings Completed!!!")

    camera = cv2.VideoCapture(0)


    while Mark:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(faces)
        encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personName[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (107, 184,250), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (107, 184,250), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if flag==0:
                    markAttendance(name)
                    flag=1

            FRAME_WINDOW.image(frame)
    else:
        st.write("Stopped")




elif a=="Display attendance sheet":

    st.title("Attendance sheet")
    df = pd.read_csv('Attendance.csv')

    st.dataframe(df)
    lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_bjyiojos.json")
    st_lottie(lottie_coding, height=300, key="marked")


