# coding: utf-8

import cv2

def generate_frame(image_size=28, destroy_callback=None):
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)

    def show_frame(frame, zoom=1, text=None, color=(255,255,255)):
        size = image_size * zoom
        frame = cv2.resize(frame, (size, size))
        if(text is not None):
            cv2.putText(
                frame,
                text,
                (3, size-3),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                color
            )
        cv2.imshow('fram', frame)

    try:
        while(True):
            ret, frame = cap.read()
            if ret == False:
                raise 'destroy'

            yield frame, show_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise 'destroy'

    except KeyboardInterrupt:
        if destroy_callback is not None:
            destroy_callback()
        cap.release()
        cv2.dstroyAllWindows()


