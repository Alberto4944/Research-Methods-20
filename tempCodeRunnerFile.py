
    #     for box in r.boxes:
    #         coordinates = (box.xyxy).tolist()[0]
    #         x1, y1, x2, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    #         box = box.xywh[0]
    #         ball_x, ball_y = float(box[0]), float(box[1])

    #         # (Left, Top) = (x, y) and (Right, Bottom) = (x, y)
    #         average = (abs(x1-x2) + abs(y1-y2) + abs(x1-y2) + abs(x2-y1)) / 4
    #         # print(f'Square Size: {average}')
    #         box = box.xywh[0]
    #         ball_x, ball_y = float(box[0]), float(box[1])
            
            
    #         # cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    #         # cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    #         cv.circle(