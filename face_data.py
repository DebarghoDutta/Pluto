import os
import cv2
import numpy as np


DATASET_PATH = "./face_dataset/"
CASCADE_FILE = "haarcascade_frontalface_alt.xml"
FACE_SIZE = (100, 100)


def get_face_cascade():
    local_path = os.path.join(os.path.dirname(__file__), CASCADE_FILE)
    cv_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)

    for path in (local_path, cv_path):
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    raise FileNotFoundError(f"Could not find {CASCADE_FILE}.")


def get_face_region(frame, x, y, w, h, offset=10):
    y_start = max(0, y - offset)
    y_end = min(frame.shape[0], y + h + offset)
    x_start = max(0, x - offset)
    x_end = min(frame.shape[1], x + w + offset)
    return frame[y_start:y_end, x_start:x_end]


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


def collect_face_data():
    os.makedirs(DATASET_PATH, exist_ok=True)

    file_name = input("Enter the name of person: ").strip()
    if not file_name:
        print("Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    face_cascade = get_face_cascade()
    skip = 0
    face_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) > 0:
            faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)

            for x, y, w, h in faces[:1]:
                face_section = get_face_region(frame, x, y, w, h, offset=10)
                face_selection = cv2.resize(face_section, FACE_SIZE)

                skip += 1
                if skip % 10 == 0:
                    face_data.append(face_selection)
                    print(f"Captured: {len(face_data)}")

                cv2.imshow("Face Section", face_selection)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Main Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if face_data:
        face_data = np.array(face_data).reshape((len(face_data), -1))
        full_path = os.path.join(DATASET_PATH, file_name + ".npy")
        np.save(full_path, face_data)
        print("Dataset shape:", face_data.shape)
        print("Dataset saved successfully at:", full_path)
    else:
        print("No faces were captured.")

    cap.release()
    cv2.destroyAllWindows()


def detect_faces():
    cap = cv2.VideoCapture(0)
    face_cascade = get_face_cascade()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for x, y, w, h in faces[:1]:
            face_offset = get_face_region(frame, x, y, w, h, offset=10)
            face_selection = cv2.resize(face_offset, FACE_SIZE)

            cv2.imshow("Face", face_selection)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_training_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset folder not found. Collect face data first.")

    face_data = []
    labels = []
    names = {}
    class_id = 0

    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith(".npy"):
            names[class_id] = file_name[:-4]
            data_item = np.load(os.path.join(DATASET_PATH, file_name))
            face_data.append(data_item)
            labels.append(class_id * np.ones((data_item.shape[0],)))
            class_id += 1

    if not face_data:
        raise ValueError("No training data found. Collect face data first.")

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    return trainset, names


def recognize_faces():
    trainset, names = load_training_data()
    cap = cv2.VideoCapture(0)
    face_cascade = get_face_cascade()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_section = get_face_region(frame, x, y, w, h, offset=5)
            face_section = cv2.resize(face_section, FACE_SIZE)

            out = knn(trainset, face_section.flatten())
            name = names[int(out)]

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("\nChoose an option:")
        print("1. Collect face data")
        print("2. Detect face")
        print("3. Recognize face")
        print("4. Exit")

        choice = input("Enter your choice: ").strip()

        try:
            match choice:
                case "1":
                    collect_face_data()
                case "2":
                    detect_faces()
                case "3":
                    recognize_faces()
                case "4":
                    print("Exiting program.")
                    break
                case _:
                    print("Invalid choice. Please select 1, 2, 3, or 4.")
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
