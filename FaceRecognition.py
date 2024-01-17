import cv2
import numpy as np
import face_recognition

# 얼굴 인코딩
image = face_recognition.load_image_file("C:\Git\python\FaceRecognition\jeonchaei.png")
face_encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [face_encoding
]
known_face_names = [
    "WARNING"
]
print('Learned encoding for', len(known_face_encodings), 'images.')


# WARNING 레이블이 붙은 얼굴 인코딩 불러오기
known_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file("C:\Git\python\FaceRecognition\jeonchaei.png"))[0]
known_face_encodings = [known_face_encoding]
known_face_names = ["WARNING"]

# 웹캠 초기화
video_capture = cv2.VideoCapture(0)


while video_capture.isOpened():
    # 웹캠에서 프레임 캡처
    ret, frame = video_capture.read()
    if ret:
    # 얼굴 위치 찾기
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) > 0:        
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # 얼굴이 미리 정의된 얼굴과 일치하는지 확인
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "UNKNOWN"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # 얼굴 주위에 상자 그리기
                if name == 'WARNING':
                    cv2.rectangle(frame, (left+5, top+10), (right+5, bottom+10), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom), (right+5, bottom+30), (0, 0, 255), cv2.FILLED)
                if name == 'UNKNOWN':
                    cv2.rectangle(frame, (left+5, top+10), (right+5, bottom+10), (255, 0, 0), 2)
                    cv2.rectangle(frame, (left, bottom), (right+5, bottom+30), (255, 0, 0), cv2.FILLED)
                    
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom+20), font, 0.6, (255, 255, 255), 1)
    else:
        print("웹캠에서 프레임을 캡처하는데 실패했습니다.")
        break
        

    # 결과 이미지 표시
    cv2.imshow('Video', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 창 닫기
video_capture.release()
cv2.destroyAllWindows()
