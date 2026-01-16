import os
import glob
import cv2
import json
import numpy as np
from ultralytics import YOLO, solutions


# SETARI

image_folder = "imagini"
output_folder = "rezultate"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# CAUTARE IMAGINI

image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
image_paths += glob.glob(os.path.join(image_folder, "*.png"))

images = []
for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_file = f"{base_name}_bounding_boxes.json"
    images.append({"img_path": img_path, "json_file": json_file})


# INCARCARE YOLO

model = YOLO("yolo11m.pt")



# FUNCTIE INTERSECTIE SIMPLA

def boxes_intersect(a, b):
    return not (
            a[2] < b[0] or
            a[0] > b[2] or
            a[3] < b[1] or
            a[1] > b[3]
    )



# PROCESARE IMAGINI

for item in images:
    img_path = item["img_path"]
    json_file = item["json_file"]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Eroare la incarcare: {img_path}")
        continue

    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    print(f"\nProcesare: {img_name}")


    #  DETECTIE MASINI YOLO (IMAGINEA 1)

    results = model(img)


    #  corectie clasificari gresite

    #for box in results[0].boxes:
        #label = model.names[int(box.cls[0])]
        #if label in ["cellphone", "book", "laptop"]:
            #box.cls[0] = model.names.index("car") 

    cars_img = results[0].plot()

    output_cars = os.path.join(output_folder, f"{base_name}_masini_detectate.jpg")
    cv2.imwrite(output_cars, cars_img)
    print(f"Salvat: {output_cars}")

    cv2.imshow(f"Masini detectate - {img_name}", cars_img)
    cv2.waitKey(0)

    
    #PARKING MANAGEMENT
    
    parking_manager = solutions.ParkingManagement(
        model="yolo11m.pt",
        json_file=json_file,
    )

    results_pm = parking_manager(img)
    parking_img = results_pm.plot_im.copy()

    # inversare culori
    red_pixels = np.all(parking_img == [0, 0, 255], axis=-1)
    green_pixels = np.all(parking_img == [0, 255, 0], axis=-1)

    parking_img[red_pixels] = [0, 255, 0]
    parking_img[green_pixels] = [0, 0, 255]


    #  DETECTARE MASINI PARCATE INCORECT

    # incarcare locuri parcare
    with open(json_file, "r") as f:
        regions = json.load(f)

    parking_boxes = []
    for region in regions:
        points = region["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        parking_boxes.append([min(xs), min(ys), max(xs), max(ys)])

    incorrect_count = 0

    # verificam doar masinile
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]

        # procesam DOAR obiectele de tip "car"
        if label != "car":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_box = [x1, y1, x2, y2]

        in_any_spot = False
        for pbox in parking_boxes:
            if boxes_intersect(car_box, pbox):
                in_any_spot = True
                break

        # masini parcate incorect 
        if not in_any_spot:
            incorrect_count += 1

            cv2.rectangle(
                parking_img,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                4
            )

            cv2.putText(
                parking_img,
                "PARCARE INCORECTA",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

    # afisare pe imagine
    if incorrect_count > 0:
        cv2.rectangle(parking_img, (10, 10), (520, 55), (0, 0, 0), -1)
        cv2.putText(
            parking_img,
            f"ATENTIE: {incorrect_count} masini parcate incorect",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            3
        )


    # SALVARE + AFISARE IMAGINEA 2

    output_parking = os.path.join(output_folder, f"{base_name}_parking.jpg")
    cv2.imwrite(output_parking, parking_img)
    print(f"Salvat: {output_parking}")

    cv2.imshow(f"Status parcare - {img_name}", parking_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nProcesare completa, toate imaginile au fost procesate")

