import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [8,12,16,20]
thumb_tip = 4
thumb_base = 2

while True:
    
    ret,img = cap.read()

    if not ret:
        print("Não foi possível capturar a imagem da câmera.")
        break

    img = cv2.flip(img,1)

    h,w,c = img.shape # shape é propriedade

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []

            for id,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            finger_fold_status = []

            for tip in tip_ids:
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)

                cv2.circle(img,(x,y),15,(203, 192, 255),cv2.FILLED)

                if lm_list[tip].x < lm_list[tip-3].x:
                    cv2.circle(img,(x,y),15,(15,216,216),cv2.FILLED)

                    finger_fold_status.append(True)

                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            thumb_tip_y = lm_list[thumb_tip].y
            thumb_base_y = lm_list[thumb_base].y

            if thumb_tip_y < thumb_base_y:
                print("Curti")
                cv2.putText(img,"Curti",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,85),3)

            elif thumb_tip_y > thumb_base_y:
                print("Não curti")
                cv2.putText(img,"Não curti",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,85),3)

            mp_drawing.draw_landmarks(img,hand_landmark,mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec((0,0,255),2,2),mp_drawing.DrawingSpec((0,255,0),4,2))

    cv2.imshow("Hands tracking",img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
hands.close()
cv2.destroyAllWindows()