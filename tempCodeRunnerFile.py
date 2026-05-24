cv.imshow("Stroke Labeler", frame)

        #  Key handling 
        key = cv.waitKey(1 if not paused else 50) & 0xFF

        if key == ord('q'):
            break